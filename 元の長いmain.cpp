#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <ctime>
#include <functional>
#include <random>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ShaderProgram.h"
#include "MeshDataTypes.h"
#include "VoxelTetrahedralizer.h"
#include "SoftBody.h"
#include "TetoMeshData.h"
#include "Sphere.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* stb_image 実装マクロは他 TU での再展開を防ぐため即座に #undef する。
   修正1+3 同時: これで CmaesUtils.h や他ヘッダ経由で stb_image.h が
   再インクルードされても stbi__err マクロ2重展開が起きない。 */
#undef  STB_IMAGE_IMPLEMENTATION
#undef  STB_IMAGE_WRITE_IMPLEMENTATION

/* Ctrl+Shift+E 軌道検索用の追加ヘッダ。stb_image 展開の"後"に置くこと。
   先頭(stb より前)に置くと include 順序が変わって stbi__err マクロの
   二重展開を引き起こしうる。 */
#include <fstream>
#include <glm/gtc/quaternion.hpp>

#include "mCutMesh.h"
#include "RegistrationUI.h"
#include "FullSphereCameraWithTarget.h"
#include "MeshDrawing.h"
#include "NoOpen3DRegistration.h"
#include "NormalCompatibleRefine.h"
#include "DepthRunner.h"
#include <filesystem>
#include "PathConfig.h"
#include "CameraAndDepth.h"

#include "UmeyamaUtils.h"
#include "DepthUtils.h"
#include "InteractionHelpers.h"
#include "FileDropHandler.h"
#include "PinholeProjection.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "RegistrationImGuiManager.h"
#include "PoseLibrary.h"
#include "ProtocolLog.h"
#include "CmaesUtils.h"
#include "CentVoxTetrahedralizerHybrid.h"
#include "AutoDeform.h"
#include "AutoHandleController.h"

CameraPreview gCameraPreview;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DepthRunner gDepthRunner;

std::vector<DepthRunnerPoint> gUserSegPoints;
std::vector<glm::vec3>        gUserSegPoints3D;
std::vector<bool>             gUserSegPointsFG;

float gDepthScale = 0.3f;
float g_voxelSize = 0.3f;
float g_idealVoxel1to1  = 0.0f;
float g_idealVoxel1to15 = 0.0f;
float g_idealVoxel1to2  = 0.0f;
static void computeIdealVoxelSizes();
const float gMeshScale = 10.0f;
int   gGridWidth = 128;

int                 gPinMode = 0;    // 0=Relief, 1=DiffPinhole, 2=PurePinhole
Pinhole::Intrinsics gLatestIntrinsics;
float               gPinholeBaseScale = 0.0f;

inline bool isPinhole() { return gPinMode != 0; }

inline bool updateLatestIntrinsicsFromOutputDir(const std::string& outputDir = "../../../depth_output")
{
    std::string p = outputDir + "/intrinsics.txt";
    return Pinhole::loadIntrinsicsTxt(p, gLatestIntrinsics);
}

inline void regenerateDepthMeshAuto(mCutMesh* mesh, float depthScale, float meshScale)
{
    if (!mesh) return;
    if (gPinMode != 0 && !gLatestIntrinsics.valid) {
        updateLatestIntrinsicsFromOutputDir();
    }

    std::cout << "\n[RegenDBG] ===== regenerateDepthMeshAuto =====" << std::endl;
    std::cout << "[RegenDBG] gPinMode=" << gPinMode << " (0=Relief 1=Diff 2=Pure)" << std::endl;
    std::cout << "[RegenDBG] depthScale=" << depthScale << " meshScale=" << meshScale << std::endl;
    std::cout << "[RegenDBG] gPinholeBaseScale=" << gPinholeBaseScale << std::endl;
    std::cout << "[RegenDBG] gLatestIntrinsics: valid=" << gLatestIntrinsics.valid
              << " fx=" << gLatestIntrinsics.fx << " fy=" << gLatestIntrinsics.fy
              << " cx=" << gLatestIntrinsics.cx << " cy=" << gLatestIntrinsics.cy
              << " w=" << gLatestIntrinsics.width << " h=" << gLatestIntrinsics.height << std::endl;
    std::cout << "[RegenDBG] mesh->depthImageData size=" << mesh->depthImageData.size() << std::endl;
    std::cout << "[RegenDBG] mesh->loadedImageWidth=" << mesh->loadedImageWidth
              << " loadedImageHeight=" << mesh->loadedImageHeight << std::endl;

    if (!mesh->depthImageData.empty()) {
        float dmin = 1e30f, dmax = -1e30f;
        int nz = 0;
        for (float v : mesh->depthImageData) {
            if (v > 0.0f) {
                if (v < dmin) dmin = v;
                if (v > dmax) dmax = v;
                nz++;
            }
        }
        std::cout << "[RegenDBG] depthImageData non-zero range: [" << dmin << ", " << dmax
                  << "] count=" << nz << "/" << mesh->depthImageData.size() << std::endl;
    }

    static int sLastMode = 0;
    if (gPinMode != sLastMode
        && mesh->loadedImageWidth > 0 && mesh->loadedImageHeight > 0) {
        bool loaded = false;
        if (gPinMode == 2) {
            // Pure Pinhole: try metric binary first
            std::string metricPath = std::string(DEPTH_OUTPUT_PATH) + "depth_metric.bin";
            if (std::filesystem::exists(metricPath)) {
                loaded = mesh->loadMetricDepth(metricPath, mesh->loadedImageWidth, mesh->loadedImageHeight);
                std::cout << "[RegenDBG] MODE CHANGED " << sLastMode << "->" << gPinMode
                          << ", loaded metric: " << metricPath << " ok=" << loaded << std::endl;
            }
            if (!loaded) {
                std::cerr << "[RegenDBG] depth_metric.bin missing, fallback to depth_masked.png" << std::endl;
                std::string depthPath = std::string(DEPTH_OUTPUT_PATH) + "depth_masked.png";
                loaded = mesh->loadDepthImage(depthPath, mesh->loadedImageWidth, mesh->loadedImageHeight);
            }
        } else {
            // Relief / Diff: 従来の PNG
            std::string depthPath = std::string(DEPTH_OUTPUT_PATH) + "depth_masked_renorm.png";
            if (std::filesystem::exists(depthPath)) {
                loaded = mesh->loadDepthImage(depthPath, mesh->loadedImageWidth, mesh->loadedImageHeight);
                std::cout << "[RegenDBG] MODE CHANGED " << sLastMode << "->" << gPinMode
                          << ", loaded PNG: " << depthPath << " ok=" << loaded << std::endl;
            }
        }
        sLastMode = gPinMode;
    } else {
        std::cout << "[RegenDBG] no mode change or data empty, sLastMode=" << sLastMode << std::endl;
    }

    regenerateDepthMesh(mesh, depthScale, meshScale,
                         gPinMode, gLatestIntrinsics, gPinholeBaseScale);

    if (mesh && !mesh->mVertices.empty()) {
        int gw = gGridWidth;
        int gh = gw * mesh->loadedImageHeight / mesh->loadedImageWidth;
        size_t nFront = (size_t)(gw + 1) * (gh + 1);
        float xMin = 1e30f, xMax = -1e30f;
        float yMin = 1e30f, yMax = -1e30f;
        float zMin = 1e30f, zMax = -1e30f;
        for (size_t i = 0; i < nFront && i*3+2 < mesh->mVertices.size(); i++) {
            float x = mesh->mVertices[i*3+0];
            float y = mesh->mVertices[i*3+1];
            float z = mesh->mVertices[i*3+2];
            if (x < xMin) xMin = x; if (x > xMax) xMax = x;
            if (y < yMin) yMin = y; if (y > yMax) yMax = y;
            if (z < zMin) zMin = z; if (z > zMax) zMax = z;
        }
        std::cout << "[RegenDBG] Front vertices after gen:" << std::endl;
        std::cout << "  X range: [" << xMin << ", " << xMax << "] width=" << (xMax-xMin) << std::endl;
        std::cout << "  Y range: [" << yMin << ", " << yMax << "] height=" << (yMax-yMin) << std::endl;
        std::cout << "  Z range: [" << zMin << ", " << zMax << "] depth=" << (zMax-zMin) << std::endl;

        size_t iTL = 0;
        size_t iTR = (size_t)gw;
        size_t iCTR = nFront / 2;
        size_t iBR = nFront - 1;
        size_t iBL = nFront - (size_t)(gw + 1);
        auto printVtx = [&](const char* name, size_t i) {
            if (i*3+2 < mesh->mVertices.size()) {
                std::cout << "  " << name << " v[" << i << "]=("
                          << mesh->mVertices[i*3+0] << ", "
                          << mesh->mVertices[i*3+1] << ", "
                          << mesh->mVertices[i*3+2] << ")" << std::endl;
            }
        };
        printVtx("TL ", iTL);
        printVtx("TR ", iTR);
        printVtx("CTR", iCTR);
        printVtx("BR ", iBR);
        printVtx("BL ", iBL);
    }
    std::cout << "[RegenDBG] ===== end =====" << std::endl << std::endl;
}

int gWindowWidth = 1280, gWindowHeight = 720;
GLFWwindow* gWindow = NULL;

std::string gDroppedFilePath = "";
bool        gFileDropped     = false;

RegistrationImGuiManager gUIManager;

glm::vec3 hit_position;
int hit_index;
bool isDragging;
float gGroupRadius = 0.5f;

glm::mat4 model(1.0), view(1.0), projection(1.0);
glm::vec3 objPos = glm::vec3(0.0f, 0.0f, 0.0f);

glm::vec3 bunnyPos = glm::vec3(0.0f, 0.0f, 0.0f);

void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode);
void glfw_OnFramebufferSize(GLFWwindow* window, int width, int height);
void glfw_onMouseMoveOrbit(GLFWwindow* window, double posX, double posY);
void glfw_onMouseScroll(GLFWwindow* window, double deltaX, double deltaY);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
bool initOpenGL();
void showFPS(GLFWwindow* window);
void setupUICallbacks();

static GLuint g_sceneTexForProgress = 0;
static bool g_sceneTexAllocated = false;

void captureSceneForProgress() {
    if (g_sceneTexForProgress == 0) {
        glGenTextures(1, &g_sceneTexForProgress);
        glBindTexture(GL_TEXTURE_2D, g_sceneTexForProgress);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    } else {
        glBindTexture(GL_TEXTURE_2D, g_sceneTexForProgress);
    }
    glReadBuffer(GL_BACK);
    if (!g_sceneTexAllocated) {
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, gWindowWidth, gWindowHeight, 0);
        g_sceneTexAllocated = true;
    } else {
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, gWindowWidth, gWindowHeight);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void showProgressOverlay(float progress, const char* message) {
    if (!gWindow) return;

    static GLuint sProg = 0, sVAO = 0, sVBO = 0;
    static GLint sLocColor = -1;

    if (sProg == 0) {
        const char* vsSrc =
            "#version 330 core\n"
            "layout(location=0) in vec2 aPos;\n"
            "void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }\n";
        const char* fsSrc =
            "#version 330 core\n"
            "uniform vec4 uColor;\n"
            "out vec4 FragColor;\n"
            "void main(){ FragColor = uColor; }\n";
        auto compile = [](GLenum type, const char* src) -> GLuint {
            GLuint s = glCreateShader(type);
            glShaderSource(s, 1, &src, nullptr);
            glCompileShader(s);
            return s;
        };
        GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
        GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
        sProg = glCreateProgram();
        glAttachShader(sProg, vs);
        glAttachShader(sProg, fs);
        glLinkProgram(sProg);
        glDeleteShader(vs);
        glDeleteShader(fs);
        sLocColor = glGetUniformLocation(sProg, "uColor");

        glGenVertexArrays(1, &sVAO);
        glGenBuffers(1, &sVBO);
        glBindVertexArray(sVAO);
        glBindBuffer(GL_ARRAY_BUFFER, sVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindVertexArray(0);
    }

    auto drawRect = [&](float px, float py, float pw, float ph, float r, float g, float b, float a) {
        float x0 = px / gWindowWidth * 2.0f - 1.0f;
        float y0 = 1.0f - py / gWindowHeight * 2.0f;
        float x1 = (px+pw) / gWindowWidth * 2.0f - 1.0f;
        float y1 = 1.0f - (py+ph) / gWindowHeight * 2.0f;
        float verts[] = { x0,y0, x1,y0, x1,y1, x0,y0, x1,y1, x0,y1 };
        glBindBuffer(GL_ARRAY_BUFFER, sVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);
        glUniform4f(sLocColor, r, g, b, a);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    };

    float barW = 380.0f, barH = 20.0f;
    float padX = 20.0f, padY = 18.0f;
    float boxW = barW + padX * 2;
    float boxH = barH + padY * 2;
    float bx = (gWindowWidth - boxW) * 0.5f;
    float by = (gWindowHeight - boxH) * 0.5f;

    GLboolean prevDepth, prevBlend;
    glGetBooleanv(GL_DEPTH_TEST, &prevDepth);
    glGetBooleanv(GL_BLEND, &prevBlend);

    static GLuint sRestoreProg = 0, sRestoreVAO = 0, sRestoreVBO = 0;
    static GLint sRestoreTexLoc = -1;
    if (sRestoreProg == 0) {
        const char* rvs =
            "#version 330 core\n"
            "layout(location=0) in vec2 aPos;\n"
            "out vec2 uv;\n"
            "void main(){ uv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }\n";
        const char* rfs =
            "#version 330 core\n"
            "uniform sampler2D tex;\n"
            "in vec2 uv;\n"
            "out vec4 FragColor;\n"
            "void main(){ FragColor = texture(tex, uv); }\n";
        auto comp = [](GLenum t, const char* s) -> GLuint {
            GLuint sh = glCreateShader(t);
            glShaderSource(sh, 1, &s, nullptr);
            glCompileShader(sh);
            return sh;
        };
        GLuint vs2 = comp(GL_VERTEX_SHADER, rvs);
        GLuint fs2 = comp(GL_FRAGMENT_SHADER, rfs);
        sRestoreProg = glCreateProgram();
        glAttachShader(sRestoreProg, vs2);
        glAttachShader(sRestoreProg, fs2);
        glLinkProgram(sRestoreProg);
        glDeleteShader(vs2);
        glDeleteShader(fs2);
        sRestoreTexLoc = glGetUniformLocation(sRestoreProg, "tex");
        float quad[] = { -1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1 };
        glGenVertexArrays(1, &sRestoreVAO);
        glGenBuffers(1, &sRestoreVBO);
        glBindVertexArray(sRestoreVAO);
        glBindBuffer(GL_ARRAY_BUFFER, sRestoreVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindVertexArray(0);
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glUseProgram(sRestoreProg);
    glBindVertexArray(sRestoreVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_sceneTexForProgress);
    glUniform1i(sRestoreTexLoc, 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(sProg);
    glBindVertexArray(sVAO);

    drawRect(bx, by, boxW, boxH, 0.08f, 0.08f, 0.12f, 0.92f);
    drawRect(bx+1, by+1, boxW-2, 1, 0.4f, 0.4f, 0.5f, 0.8f);
    drawRect(bx+1, by+boxH-2, boxW-2, 1, 0.4f, 0.4f, 0.5f, 0.8f);
    drawRect(bx, by, 1, boxH, 0.4f, 0.4f, 0.5f, 0.8f);
    drawRect(bx+boxW-1, by, 1, boxH, 0.4f, 0.4f, 0.5f, 0.8f);
    drawRect(bx+padX, by+padY, barW, barH, 0.15f, 0.15f, 0.2f, 1.0f);
    float fillW = barW * glm::clamp(progress, 0.0f, 1.0f);
    if (fillW > 0)
        drawRect(bx+padX, by+padY, fillW, barH, 0.2f, 0.7f, 0.4f, 1.0f);

    glBindVertexArray(0);
    glUseProgram(0);
    if (prevDepth) glEnable(GL_DEPTH_TEST);
    if (!prevBlend) glDisable(GL_BLEND);

    glfwSwapBuffers(gWindow);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glUseProgram(sRestoreProg);
    glBindVertexArray(sRestoreVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_sceneTexForProgress);
    glUniform1i(sRestoreTexLoc, 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glUseProgram(0);
    if (prevDepth) glEnable(GL_DEPTH_TEST);
    if (!prevBlend) glDisable(GL_BLEND);

}

FullSphereCamera OrbitCam;

class Grabber {
public:
    Grabber() :
        physicsObject(nullptr),
        grabDistance(0.0f),
        prevPosition(0.0f),
        velocity(0.0f),
        time(0.0f)
    {}

    void setPhysicsObject(SoftBody* object) {
        physicsObject = object;
    }

    void startGrab(float screenX, float screenY) {
        if (!physicsObject) return;

        RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection,
                                                     glm::vec4(0, 0, gWindowWidth, gWindowHeight));

        glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);

        RayCast::Ray localRay;
        localRay.origin = glm::vec3(invModelMatrix * glm::vec4(worldRay.origin, 1.0f));
        localRay.direction = glm::normalize(glm::vec3(invModelMatrix * glm::vec4(worldRay.direction, 0.0f)));

        RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);

        if (hit.hit) {
            glm::vec4 worldHitPos = modelMatrix * glm::vec4(hit.position, 1.0f);
            hit_position = glm::vec3(worldHitPos);
            grabDistance = glm::length(hit_position - worldRay.origin);
            prevPosition = hit_position;
            velocity = glm::vec3(0.0f);
            time = 0.0f;

            float grabThreshold = 1.0f;
            if (physicsObject && !physicsObject->handleGroups.empty()) {
                for (const auto& g : physicsObject->handleGroups)
                    grabThreshold = std::max(grabThreshold, g.radius);
            }
            physicsObject->smartGrab(hit.position, grabThreshold);
            isDragging = true;
        }
    }

    bool hitTest(float screenX, float screenY) {
        if (!physicsObject) return false;
        RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection,
                                                     glm::vec4(0, 0, gWindowWidth, gWindowHeight));
        glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);
        RayCast::Ray localRay;
        localRay.origin = glm::vec3(invModelMatrix * glm::vec4(worldRay.origin, 1.0f));
        localRay.direction = glm::normalize(glm::vec3(invModelMatrix * glm::vec4(worldRay.direction, 0.0f)));
        RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);
        return hit.hit;
    }

    void moveGrab(float screenX, float screenY, float deltaTime) {
        if (!physicsObject || !isDragging) return;

        RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection,
                                                     glm::vec4(0, 0, gWindowWidth, gWindowHeight));

        glm::vec3 newPosition = worldRay.origin + worldRay.direction * grabDistance;

        if (time > 0.0f) {
            velocity = (newPosition - prevPosition) / time;
        }

        hit_position = newPosition;

        glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);
        glm::vec3 localPos = glm::vec3(invModelMatrix * glm::vec4(newPosition, 1.0f));
        glm::vec3 localVel = glm::vec3(invModelMatrix * glm::vec4(velocity, 0.0f));

        physicsObject->smartMove(localPos, localVel);

        prevPosition = newPosition;
        time = deltaTime;
    }

    void endGrab() {
        if (physicsObject) {
            glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), bunnyPos);
            glm::mat4 invModelMatrix = glm::inverse(modelMatrix);
            glm::vec3 localPos = glm::vec3(invModelMatrix * glm::vec4(hit_position, 1.0f));
            glm::vec3 localVel = glm::vec3(invModelMatrix * glm::vec4(velocity, 0.0f));

            physicsObject->smartEndGrab(localPos, localVel);
        }
        isDragging = false;
    }

    void placeSphere(float screenX, float screenY, float groupRadius = 1.0f) {
        if (!physicsObject) return;

        RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection,
                                                     glm::vec4(0, 0, gWindowWidth, gWindowHeight));

        glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);

        RayCast::Ray localRay;
        localRay.origin = glm::vec3(invModelMatrix * glm::vec4(worldRay.origin, 1.0f));
        localRay.direction = glm::normalize(glm::vec3(invModelMatrix * glm::vec4(worldRay.direction, 0.0f)));

        RayCast::RayHit hit = RayCast::intersectMesh(localRay, *physicsObject);

        if (hit.hit) {

            physicsObject->createHandleGroup(hit.position, groupRadius);

            glm::vec4 worldHitPos = modelMatrix * glm::vec4(hit.position, 1.0f);
            hit_position = glm::vec3(worldHitPos);
            hit_index = physicsObject->handleGroups.size() - 1;

            std::cout << "Created handle group with radius " << groupRadius
                      << " at position: " << hit.position.x << ", "
                      << hit.position.y << ", " << hit.position.z << std::endl;
        } else {
            hit_index = -1;
        }

        isDragging = false;
    }

    void update(float deltaTime) {
        time += deltaTime;
    }

private:
    SoftBody* physicsObject;
    float grabDistance;
    glm::vec3 prevPosition;
    glm::vec3 velocity;
    float time;
};

Grabber* gGrabber = nullptr;

SoftBody *multiBody;

constexpr int DEFAULT_GRID_SIZE = 10;

struct DeformHandlPlaceData {
    enum State {
        RIGID_MODE,
        HANDLE_PLACE_MODE,
        DEFORM_MODE,
        PLANECUT_MODE
    };

    State state = RIGID_MODE;
    std::vector<glm::vec3> softbodyPoints;

    void reset() {
        softbodyPoints.clear();
        state = RIGID_MODE;

        if (multiBody) {
            multiBody->fullReset();
        }

        std::cout << "HandlePlace data reset with full mesh restoration" << std::endl;
    }
};

RegistrationData registrationHandle;
NormalRefine::RefineState g_refineState;
PoseLibrary g_poseLibrary;

/* ------------------------------------------------------------
 * Shift+P protocol infrastructure (Step 4a)
 * ------------------------------------------------------------
 * g_protocolLog       : write-only logger for protocol runs
 *                       (independent from g_poseLibrary)
 * g_suppressPoseLibSave : when true, poseSaveToLibrary() is a no-op.
 *                       The protocol runner sets this to true during
 *                       its automated phases so that reused HemiAuto /
 *                       CMA-ES / Refine code paths do not pollute the
 *                       interactive pose library with 130 entries per
 *                       image. Default is false — UI behavior unchanged.
 * ------------------------------------------------------------ */
ProtocolLog g_protocolLog;
bool        g_suppressPoseLibSave = false;

std::vector<size_t> g_refineVertexIndices;
DeformHandlPlaceData deformHandlPlace;
SphereMesh deformSphereMarker;
SphereMesh registrationSphereMarker;

enum MainMode {
    REGISTRATION_MODE,
    DEFORM_MODE
};

mCutMesh *screenMesh;
mCutMesh *arMesh;

inline int gGridHeight() {
    if (screenMesh && screenMesh->loadedImageWidth > 0 && screenMesh->loadedImageHeight > 0)
        return gGridWidth * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
    return gGridWidth * 9 / 16;
}

inline void resetBoundaryMap() {
    g_boundaryDistMap.valid = false;
    std::string maskPath = DEPTH_OUTPUT_PATH + std::string("segmentation_mask.png");
    if (std::filesystem::exists(maskPath))
        loadMaskAndComputeBoundaryMap(maskPath);
}

MainMode currentMainMode = REGISTRATION_MODE;

bool saveARimage = false;
bool saveARimageOrtho = false;   // Shift+A: ortho projection variant (silhouette comparison)
bool deformInit = false;

GLuint g_arPreviewTex = 0;
bool g_showARPreview = false;
int g_arPreviewW = 0;
int g_arPreviewH = 0;
std::string g_arPreviewPath;

std::function<void(float, const char*)> g_progressCallback = nullptr;

std::vector<glm::vec3> g_cluster1Points;
std::vector<glm::vec3> g_cluster2Points;
std::vector<glm::vec3> g_targetPoints;
bool g_showClusterVisualization = false;
bool g_showCorrespondencePoints = false;

AutoDeform::State gAutoDeform;
static AutoHandleController gAutoCtrl;
static int gAutoDeformPresetIdx = 0;  /* 0=P0 Sequential, 1=P1 Conservative, 2=P2 Standard, 3=P3 Aggressive, 4=P4 Extreme */
static int gAutoDeformDebugFrames = 0;
static std::vector<float> gAutoDeformDebugBaseline;
static float gMoveScale = 0.1f;
static int   gPendingRmseFrames      = 0;
static float gPendingRmseImmediate   = -1.0f;
static float gAutoDeformBoostDamping = 0.3f;
static int   gAutoDeformBoostIter    = 30;
static bool  gAutoDeformFirstCommit  = false;
/* AutoDeform before/after snapshots */
static std::vector<float> gSnapBeforePositions;
static std::vector<std::vector<float>> gSnapBeforeVisPositions;
static std::vector<float> gSnapAfterPositions;
static std::vector<std::vector<float>> gSnapAfterVisPositions;
static bool gSnapBeforeValid = false;
static bool gSnapAfterValid  = false;
static bool gShowingAfter    = true;  /* true=after state, false=before state */

static std::string g_currentOrientLabel   = "Front";
static int         g_currentOrientRunCount = 0;

static float                             g_bestSessionCompRmse   = FLT_MAX;
static float                             g_bestSessionIoU2D      = 0.0f;
static std::vector<std::vector<GLfloat>> g_bestSessionVertices;
static std::vector<std::vector<GLfloat>> g_bestSessionNormals;

static int  g_sessionId          = 1;
static int  g_sessionBipopN      = 0;
static int  g_sessionRefineN     = 0;
static int  g_sessionSilhouetteN = 0;
static std::chrono::steady_clock::time_point g_stepStartTime = std::chrono::steady_clock::now();

/* poseSaveToLibrary() の採択基準。
   RMSE   : compRmse が session-best 以下なら accept (従来動作。HemiAuto/BIPOP/Refine/Umeyama)
   IOU    : compIoU2D が session-best より大きければ accept (Shift+E)
   EITHER : どちらかが改善すれば accept (将来拡張用) */
enum class SaveCriterion { RMSE, IOU, EITHER };

mCutMesh *liverMesh3D;
mCutMesh *gbMesh3D;
mCutMesh *portalMesh3D;
mCutMesh *veinMesh3D;
mCutMesh *tumorMesh3D;
mCutMesh *segmentMesh3D;

std::vector<mCutMesh*> allMeshes;

std::vector<float> meshAlphaValues = {
    0.8f,
    0.9f,
    0.9f,
    0.9f,
    0.5f,
    0.5f,
    0.7f
};

bool splitScreenMode = false;
bool depthSplitScreenMode = false;
FullSphereCamera OrbitCamLeft_Target;
FullSphereCamera OrbitCamRight_Screen;

bool cameraUse = false;

mCutMesh *cutterMesh = nullptr;

float scaleSpeed = 1.1;

// -------------------------------------------------------
// Pose Library helpers
// -------------------------------------------------------
static std::vector<mCutMesh*> getOrganList() {
    return {liverMesh3D, portalMesh3D, veinMesh3D,
            tumorMesh3D, segmentMesh3D, gbMesh3D};
}

static std::vector<std::vector<GLfloat>> g_initOrganVertices;
static std::vector<std::vector<GLfloat>> g_initOrganNormals;

static void snapshotInitialPose() {
    auto organs = getOrganList();
    g_initOrganVertices.resize(organs.size());
    g_initOrganNormals.resize(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        g_initOrganVertices[i] = organs[i]->mVertices;
        g_initOrganNormals[i]  = organs[i]->mNormals;
    }
    std::cout << "[PoseLibrary] Initial pose snapshot saved." << std::endl;
}

static glm::mat4 computeCurrentTransform() {
    if (g_initOrganVertices.empty()) return glm::mat4(1.0f);
    return PoseLibrary::computeTransformFromLiver(
        g_initOrganVertices[0], liverMesh3D->mVertices);
}

/* ---- Shift+T (Ctrl+Shift+E) 軌道検索用: Shift+V / Shift+E 終了時に姿勢を自動保存 ----
   軌道検索は P_3D (BIPOP=Shift+V 解) と P_2D (2D-BIPOP=Shift+E 解) の間を
   SLERP/lerp/log-lerp で補間する。端点は Shift+V / Shift+E の成功完了時に
   その時の SRT 変換 (computeTransformFromLiver で初期姿勢から算出) を保存。
   Pose Library への保存とは独立 (Shift+E は library に保存しない実験モードのため)。 */
static glm::mat4 gShiftV_lastTransform = glm::mat4(1.0f);
static bool      gShiftV_lastValid     = false;
static glm::mat4 gShiftE_lastTransform = glm::mat4(1.0f);
static bool      gShiftE_lastValid     = false;

/* Ctrl+Shift+E 軌道スクリーンショット用グローバル (main() でポインタセット)。
   key handler から描画パスを呼ぶ必要があるため、メインローカルのシェーダを
   参照する手段として。 */
static ShaderProgram* g_shaderProgram     = nullptr;
static ShaderProgram* g_shaderProgramCube = nullptr;

/* 軌道の各 t で現在のカメラから見た scene を PNG に保存するヘルパー。
   2 枚の PNG を同じカメラ・viewport で出力:
     t_XX.png       overlay  — 肝臓 + 全 organ + screenMesh の AR 合成 (論文 Figure 用)
     t_XX_seg8.png  mask     — seg8 bump を白、他は黒 (binary, IoU 計算用)
   なお、物理画像 (screenMesh のみ) の capture は trajectory 全体で 1 回だけ行う
   ため、この関数の外側 (Ctrl+Shift+E ハンドラ) で別途実施する。
   FBO を毎回作り直す (軽量なので問題なし、11 点で合計 22 回描画)。
   注意: このヘルパーは glfwPollEvents の key callback 中から呼ばれる。
   GL コンテキストは active なので直接描画可能。 */
static void saveTrajectoryFrame(const std::string& outDir, float t_val,
                                const glm::mat4& viewMat, const glm::mat4& projMat,
                                const glm::mat4& modelMat, const glm::vec3& camPos)
{
    if (!g_shaderProgram || !g_shaderProgramCube) return;
    if (!liverMesh3D) return;

    const int W = gWindowWidth;
    const int H = gWindowHeight;

    GLuint fbo = 0, colorTex = 0, depthRbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, colorTex, 0);

    glGenRenderbuffers(1, &depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, depthRbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteTextures(1, &colorTex);
        glDeleteRenderbuffers(1, &depthRbo);
        glDeleteFramebuffers(1, &fbo);
        std::cerr << "[TrajFrame] FBO not complete, skipping screenshot" << std::endl;
        return;
    }

    /* GLviewport/状態の退避 */
    GLint prevViewport[4];
    glGetIntegerv(GL_VIEWPORT, prevViewport);

    char tbuf[32];
    snprintf(tbuf, sizeof(tbuf), "%.2f", t_val);

    /* ---- Frame 1: overlay (全 organ + screenMesh) ---- */
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, W, H);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        std::vector<glm::vec4> arColors = {
            glm::vec4(0.8f, 0.2f, 0.2f, 0.80f),  /* liver (red) */
            glm::vec4(0.9f, 0.6f, 0.6f, 0.90f),  /* portal */
            glm::vec4(0.2f, 0.8f, 0.8f, 0.90f),  /* vein */
            glm::vec4(0.8f, 0.2f, 0.8f, 0.90f),  /* tumor */
            glm::vec4(0.8f, 0.8f, 0.0f, 0.60f),  /* segment (yellow) */
            glm::vec4(0.2f, 0.5f, 0.2f, 0.60f),  /* gb */
            glm::vec4(1.0f, 1.0f, 1.0f, 0.70f)   /* screen (with texture slot 6) */
        };
        std::vector<mCutMesh*> arMeshes = {
            liverMesh3D, portalMesh3D, veinMesh3D,
            tumorMesh3D, segmentMesh3D, gbMesh3D, screenMesh
        };
        draw_AllmCutMeshes(arMeshes, *g_shaderProgram, *g_shaderProgramCube,
                           camPos, arColors, modelMat, viewMat, projMat, 6);

        std::vector<unsigned char> pixels(W * H * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);   /* avoid row padding for non-mod-4 widths */
        glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        int stride = W * 3;
        std::vector<unsigned char> flipped(pixels.size());
        for (int y = 0; y < H; y++)
            memcpy(&flipped[y * stride], &pixels[(H - 1 - y) * stride], stride);

        std::string path = outDir + "/t_" + tbuf + ".png";
        stbi_write_png(path.c_str(), W, H, 3, flipped.data(), stride);
    }

    /* ---- Frame 2: segment 8 (shrunken-liver depth occlusion) --------------
       目的: 物理 phantom の「肝臓表面に出ている seg8 の盛り上がり」と幾何的に
            一致する mask を得る。

       手法 (確定版):
         1. liver mesh を「重心ベースに 0.99 倍に縮小」した model 行列で黒く描画
            → depth buffer に「liver より 1% 内側の表面」が記録される
         2. seg8 を不透明黄色で GL_LEQUAL 描画
            - bump (liver 表面の外側) → 縮小 liver より手前 → 黄色描画 ✓
            - cut surface (liver 内部) → 縮小 liver の中 → 黒に occlude ✓
         3. 縮小は GPU 側 model 行列で実現するので mesh data は touch しない

       Z-fighting 完全排除: depth 差は (liver 半径 × 1%) ~ mm スケール、
                          z-buffer 精度 (μm) より遥か上。

       Silhouette inflation: bump 縁が liver 表面の 1% 外側まで広がるが、
                            これは原理的に避けられない (cut を hide するために
                            縮小は必要)。SHRINK_FACTOR で trade-off 可能:
                              0.99   → 縁 1% 外側膨張、cut hide 確実
                              0.995  → 縁 0.5% 外側膨張、cut が浅いと露出リスク
                              0.999  → 縁 0.1% 膨張、cut 露出リスクあり ----------*/
    if (segmentMesh3D) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, W, H);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* ===== Liver 重心 (現フレームの deformed 状態) ===== */
        const auto& LV = liverMesh3D->mVertices;
        const size_t numLiverVerts = LV.size() / 3;
        glm::vec3 liverCentroid(0.0f);
        for (size_t i = 0; i < numLiverVerts; i++) {
            liverCentroid += glm::vec3(LV[i*3], LV[i*3+1], LV[i*3+2]);
        }
        liverCentroid /= (float)numLiverVerts;

        /* ===== 縮小行列 (object space で重心中心の 0.99 倍 scale) ===== */
        const float SHRINK_FACTOR = 0.99f;
        glm::mat4 shrinkMat = glm::translate(glm::mat4(1.0f), liverCentroid)
                            * glm::scale(glm::mat4(1.0f), glm::vec3(SHRINK_FACTOR))
                            * glm::translate(glm::mat4(1.0f), -liverCentroid);
        glm::mat4 liverModelMat = modelMat * shrinkMat;

        /* GL state を退避 */
        GLboolean prevDepthTest = glIsEnabled(GL_DEPTH_TEST);
        GLboolean prevCullFace  = glIsEnabled(GL_CULL_FACE);
        GLboolean prevBlend     = glIsEnabled(GL_BLEND);
        GLint     prevDepthFunc; glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);
        GLint     prevCullMode;  glGetIntegerv(GL_CULL_FACE_MODE, &prevCullMode);
        GLboolean prevDepthMask; glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);

        g_shaderProgram->use();
        g_shaderProgram->setUniform("view",       viewMat);
        g_shaderProgram->setUniform("projection", projMat);
        g_shaderProgram->setUniform("lightPos",   camPos);
        g_shaderProgram->setUniform("viewPos",    camPos);
        g_shaderProgram->setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
        g_shaderProgram->setUniform("useTexture", 0);

        /* ---- Pass 1: 縮小 liver を黒で描画 (color + depth) ---- */
        glDepthFunc(GL_LESS);
        g_shaderProgram->setUniform("model",     liverModelMat);
        g_shaderProgram->setUniform("vertColor", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        glBindVertexArray(liverMesh3D->VAO);
        glDrawElements(GL_TRIANGLES,
                       (GLsizei)liverMesh3D->mIndices.size(),
                       GL_UNSIGNED_INT, 0);

        /* ---- Pass 2: seg8 を白で描画 (GL_LEQUAL, 元の modelMat) ---- */
        glDepthFunc(GL_LEQUAL);
        g_shaderProgram->setUniform("model",     modelMat);
        g_shaderProgram->setUniform("vertColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        glBindVertexArray(segmentMesh3D->VAO);
        glDrawElements(GL_TRIANGLES,
                       (GLsizei)segmentMesh3D->mIndices.size(),
                       GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);

        /* GL state 復元 */
        glDepthMask(prevDepthMask);
        glDepthFunc(prevDepthFunc);
        if (prevCullFace) { glEnable(GL_CULL_FACE);  glCullFace(prevCullMode); }
        else              { glDisable(GL_CULL_FACE); }
        if (prevBlend)      glEnable(GL_BLEND);  else glDisable(GL_BLEND);
        if (!prevDepthTest) glDisable(GL_DEPTH_TEST);

        /* PNG 保存 (pure binary 化: 輝度 > 0 は全て白、それ以外は黒) */
        std::vector<unsigned char> pixels(W * H * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);   /* avoid row padding for non-mod-4 widths */
        glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        /* 二値化: R+G+B が 0 より大きければ (255,255,255)、そうでなければ (0,0,0) */
        for (size_t i = 0; i < pixels.size(); i += 3) {
            int sum = pixels[i] + pixels[i+1] + pixels[i+2];
            unsigned char v = (sum > 0) ? 255 : 0;
            pixels[i] = pixels[i+1] = pixels[i+2] = v;
        }

        int stride = W * 3;
        std::vector<unsigned char> flipped(pixels.size());
        for (int y = 0; y < H; y++)
            memcpy(&flipped[y * stride], &pixels[(H - 1 - y) * stride], stride);

        std::string path = outDir + "/t_" + tbuf + "_seg8.png";
        stbi_write_png(path.c_str(), W, H, 3, flipped.data(), stride);

        std::cout << "[TrajFrame seg8] t=" << tbuf
                  << " liverVerts=" << numLiverVerts
                  << " liverCentroid=(" << liverCentroid.x << ","
                  << liverCentroid.y << "," << liverCentroid.z << ")"
                  << " shrink=" << SHRINK_FACTOR << std::endl;
    }

    /* 状態復元 */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    glDeleteTextures(1, &colorTex);
    glDeleteRenderbuffers(1, &depthRbo);
    glDeleteFramebuffers(1, &fbo);
}

/* ============================================================
 * buildProjectedLiverMask  —  camera-aware target for Shift+E
 * ============================================================
 * INSTRUMENTED VERSION: prints exhaustive diagnostics at every step
 * and checks GL errors after every GL call so we can pinpoint the
 * exact line that crashes.
 *
 * Bug under investigation:
 *   HemiAuto → Shift+E   crashes
 *   HemiAuto → Shift+V → Shift+E   works
 * Hypothesis pool:
 *   (A) Stale view/projection passed in (HemiAuto blocks main loop;
 *       OrbitCam.UpdateCamera() doesn't run; globals stale)
 *   (B) screenMesh VAO/VBO desynced after HemiAuto's GL state churn
 *   (C) FBO incompatible with current default-FB format
 *   (D) draw_AllmCutMeshes traverses an organ list / mesh in an
 *       unexpected state and segfaults
 * ============================================================ */
#define PLM_GL_CHECK(label) do {                                        \
    GLenum _e;                                                          \
    while ((_e = glGetError()) != GL_NO_ERROR) {                        \
        std::cerr << "[ProjectedLiverMask][GL ERROR @ " << label        \
                  << "] 0x" << std::hex << _e << std::dec << std::endl; \
    }                                                                   \
} while (0)

static void buildProjectedLiverMask(const glm::mat4& viewMat,
                                    const glm::mat4& projMat,
                                    const glm::mat4& modelMat,
                                    const glm::vec3& camPos)
{
    std::cout << "[ProjectedLiverMask] === ENTER ===" << std::endl;
    PLM_GL_CHECK("entry (errors leaked from prior code)");

    g_projectedLiverMask.invalidate();

    /* ---- A. precondition checks ---- */
    if (!g_shaderProgram || !g_shaderProgramCube) {
        std::cerr << "[ProjectedLiverMask] FATAL: shaders null  "
                  << "g_shaderProgram=" << (void*)g_shaderProgram
                  << "  g_shaderProgramCube=" << (void*)g_shaderProgramCube
                  << std::endl;
        return;
    }
    if (!screenMesh) {
        std::cerr << "[ProjectedLiverMask] FATAL: screenMesh null" << std::endl;
        return;
    }
    if (!g_boundaryDistMap.valid) {
        std::cerr << "[ProjectedLiverMask] FATAL: g_boundaryDistMap invalid; "
                  << "run depth/SAM2 first" << std::endl;
        return;
    }

    /* ---- B. screenMesh integrity ---- */
    std::cout << "[ProjectedLiverMask] screenMesh: VAO=" << screenMesh->VAO
              << "  VBO=" << screenMesh->VBO
              << "  EBO=" << screenMesh->EBO
              << "  vertices=" << screenMesh->mVertices.size() / 3
              << "  indices=" << screenMesh->mIndices.size()
              << "  textureID=" << screenMesh->textureID
              << "  hasTexture=" << (screenMesh->hasTexture ? 1 : 0)
              << std::endl;

    if (screenMesh->VAO == 0) {
        std::cerr << "[ProjectedLiverMask] FATAL: screenMesh->VAO == 0; "
                  << "setUp() not called or mesh corrupted" << std::endl;
        return;
    }
    if (screenMesh->mIndices.empty() || screenMesh->mVertices.empty()) {
        std::cerr << "[ProjectedLiverMask] FATAL: screenMesh has no geometry"
                  << std::endl;
        return;
    }

    /* ---- C. matrix sanity ---- */
    auto matFinite = [](const glm::mat4& m) {
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            if (!std::isfinite(m[i][j])) return false;
        return true;
    };
    auto matIsIdentity = [](const glm::mat4& m) {
        return m == glm::mat4(1.0f);
    };
    auto matIsZero = [](const glm::mat4& m) {
        return m == glm::mat4(0.0f);
    };

    std::cout << "[ProjectedLiverMask] camPos=("
              << camPos.x << "," << camPos.y << "," << camPos.z << ")"
              << std::endl;
    std::cout << "[ProjectedLiverMask] view  finite=" << matFinite(viewMat)
              << " identity=" << matIsIdentity(viewMat)
              << " zero=" << matIsZero(viewMat) << std::endl;
    std::cout << "[ProjectedLiverMask] proj  finite=" << matFinite(projMat)
              << " identity=" << matIsIdentity(projMat)
              << " zero=" << matIsZero(projMat) << std::endl;
    std::cout << "[ProjectedLiverMask] model finite=" << matFinite(modelMat)
              << " identity=" << matIsIdentity(modelMat)
              << " zero=" << matIsZero(modelMat) << std::endl;

    if (!matFinite(viewMat) || !matFinite(projMat) || !matFinite(modelMat)) {
        std::cerr << "[ProjectedLiverMask] FATAL: non-finite matrix; "
                  << "aborting before GL ops" << std::endl;
        return;
    }
    if (matIsIdentity(projMat) || matIsZero(projMat)) {
        std::cerr << "[ProjectedLiverMask] FATAL: projection is identity/zero; "
                  << "OrbitCam not yet UpdateCamera()-ed. Aborting." << std::endl;
        return;
    }

    const int W = gWindowWidth;
    const int H = gWindowHeight;
    std::cout << "[ProjectedLiverMask] window=" << W << "x" << H << std::endl;
    if (W <= 0 || H <= 0 || W > 16384 || H > 16384) {
        std::cerr << "[ProjectedLiverMask] FATAL: bad window size" << std::endl;
        return;
    }

    /* ---- D. FBO 作成 ---- */
    std::cout << "[ProjectedLiverMask] step 1: create FBO" << std::endl;
    GLuint fbo = 0, colorTex = 0, depthRbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    PLM_GL_CHECK("after glGenFramebuffers/Bind");

    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, colorTex, 0);
    PLM_GL_CHECK("after color attachment");

    glGenRenderbuffers(1, &depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, depthRbo);
    PLM_GL_CHECK("after depth attachment");

    GLint prevViewport[4];
    glGetIntegerv(GL_VIEWPORT, prevViewport);
    GLint prevFBO = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFBO);
    std::cout << "[ProjectedLiverMask] prevViewport="
              << prevViewport[0] << "," << prevViewport[1] << ","
              << prevViewport[2] << "," << prevViewport[3]
              << "  prevFBO=" << prevFBO << std::endl;

    GLenum fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fbStatus != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[ProjectedLiverMask] FATAL: FBO not complete; status=0x"
                  << std::hex << fbStatus << std::dec << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);
        glDeleteTextures(1, &colorTex);
        glDeleteRenderbuffers(1, &depthRbo);
        glDeleteFramebuffers(1, &fbo);
        return;
    }

    /* ---- E. mask texture upload ---- */
    std::cout << "[ProjectedLiverMask] step 2: upload mask texture" << std::endl;
    const int mw = g_boundaryDistMap.width;
    const int mh = g_boundaryDistMap.height;
    std::cout << "[ProjectedLiverMask] mask size=" << mw << "x" << mh << std::endl;
    std::vector<unsigned char> maskRGB(mw * mh * 3);
    int maskInsidePx = 0;
    for (int y = 0; y < mh; y++) {
        int srcY = mh - 1 - y;
        for (int x = 0; x < mw; x++) {
            unsigned char v =
                (g_boundaryDistMap.data[srcY * mw + x] < 9000.0f) ? 255 : 0;
            int di = (y * mw + x) * 3;
            maskRGB[di] = maskRGB[di+1] = maskRGB[di+2] = v;
            if (v) maskInsidePx++;
        }
    }
    std::cout << "[ProjectedLiverMask] mask inside px=" << maskInsidePx
              << " (" << (100.0 * maskInsidePx / (mw*mh)) << "%)" << std::endl;

    GLuint maskTex = 0;
    glGenTextures(1, &maskTex);
    glBindTexture(GL_TEXTURE_2D, maskTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mw, mh, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, maskRGB.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    PLM_GL_CHECK("after mask texture upload");

    /* ---- F. swap screenMesh texture ---- */
    std::cout << "[ProjectedLiverMask] step 3: swap screenMesh texture "
              << "orig=" << screenMesh->textureID
              << " hasTex=" << (screenMesh->hasTexture ? 1 : 0)
              << " -> mask=" << maskTex << std::endl;
    const GLuint origTexId  = screenMesh->textureID;
    const bool   origHasTex = screenMesh->hasTexture;
    screenMesh->textureID  = maskTex;
    screenMesh->hasTexture = true;

    /* ---- G. set GL state and clear ---- */
    std::cout << "[ProjectedLiverMask] step 4: set GL state" << std::endl;
    glViewport(0, 0, W, H);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    PLM_GL_CHECK("after GL state set");

    /* ---- H. draw screenMesh through mask texture ---- */
    std::cout << "[ProjectedLiverMask] step 5: draw screenMesh "
              << "(triangles=" << (screenMesh->mIndices.size() / 3)
              << ")" << std::endl;
    std::vector<mCutMesh*> imgMeshes = { screenMesh };
    std::vector<glm::vec4> imgColors = { glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
    draw_AllmCutMeshes(imgMeshes, *g_shaderProgram, *g_shaderProgramCube,
                       camPos, imgColors, modelMat, viewMat, projMat, 0);
    PLM_GL_CHECK("after draw_AllmCutMeshes");
    std::cout << "[ProjectedLiverMask] step 5 done" << std::endl;

    /* ---- I. read back ----
       CRITICAL: GL_PACK_ALIGNMENT defaults to 4. With RGB (3 bytes/pixel) and
       a window whose width is not a multiple of 4 (e.g. 1850 -> 1850*3 = 5550,
       not divisible by 4), OpenGL pads each row to the next 4-byte boundary
       and writes BEYOND our W*H*3 buffer -> heap corruption -> crash.
       Fix: force byte-tight packing. Must be set before glReadPixels. */
    std::cout << "[ProjectedLiverMask] step 6: glReadPixels" << std::endl;
    std::vector<unsigned char> pixels(W * H * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    PLM_GL_CHECK("after glReadPixels");

    g_projectedLiverMask.data.assign(W * H, 0);
    int insideCount = 0;
    for (int y = 0; y < H; y++) {
        int srcY = H - 1 - y;
        for (int x = 0; x < W; x++) {
            int si = (srcY * W + x) * 3;
            int sum = (int)pixels[si] + (int)pixels[si+1] + (int)pixels[si+2];
            uint8_t v = (sum > 192) ? 255 : 0;
            g_projectedLiverMask.data[y * W + x] = v;
            if (v) insideCount++;
        }
    }
    g_projectedLiverMask.width  = W;
    g_projectedLiverMask.height = H;
    g_projectedLiverMask.valid  = true;

    /* ---- J. restore ---- */
    std::cout << "[ProjectedLiverMask] step 7: restore" << std::endl;
    screenMesh->textureID  = origTexId;
    screenMesh->hasTexture = origHasTex;
    glDeleteTextures(1, &maskTex);

    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);
    glDeleteTextures(1, &colorTex);
    glDeleteRenderbuffers(1, &depthRbo);
    glDeleteFramebuffers(1, &fbo);
    PLM_GL_CHECK("after cleanup");

    std::cout << "[ProjectedLiverMask] === EXIT === built " << W << "x" << H
              << "  inside_px=" << insideCount
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * insideCount / (double)(W * H)) << "%)"
              << std::defaultfloat << std::endl;
}

/* trajectory 全体で 1 枚だけ、現カメラから screenMesh のみを描画した物理画像を
   保存するヘルパー。
   - depth slider (gDepthScale) を一時的に 0.0 にし (screen mesh 平面化)
     regenerateDepthMesh で再生成 → capture → 元の depth scale で再生成復元
   - 他の臓器は一切描かない → pure な物理画像
   - 出力:
       {outDir}/image.png       通常の物理画像 (texture = 元画像)
       {outDir}/liver_mask.png  screenMesh の texture を一時的に SAM2 mask
                                (g_boundaryDistMap から生成) に差し替えて同じ
                                カメラで撮影。image.png と pixel 単位で一致する
                                liver silhouette GT。
   注意: regenerateDepthMesh は ~100ms-1s の重い処理だが、trajectory 全体で
        2 回 (平面化 + 復元) のみなので許容範囲。 */
static void saveTrajectoryImageOnce(const std::string& outDir,
                                    const glm::mat4& viewMat,
                                    const glm::mat4& projMat,
                                    const glm::mat4& modelMat,
                                    const glm::vec3& camPos)
{
    if (!g_shaderProgram || !g_shaderProgramCube) return;
    if (!screenMesh) return;

    /* 1. depth slider を退避して 0 に、mesh を平面化 */
    const float origDepthScale = gDepthScale;
    std::cout << "[TrajImage] depth slider: " << origDepthScale
              << " -> 0 (flattening screen mesh)" << std::endl;
    regenerateDepthMeshAuto(screenMesh, 0.0f, gMeshScale);

    /* 2. FBO を作って capture */
    const int W = gWindowWidth;
    const int H = gWindowHeight;

    GLuint fbo = 0, colorTex = 0, depthRbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, colorTex, 0);

    glGenRenderbuffers(1, &depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, depthRbo);

    GLint prevViewport[4];
    glGetIntegerv(GL_VIEWPORT, prevViewport);

    auto captureScreenMeshToPng = [&](const std::string& outPath, bool binarize) {
        /* 現在の screenMesh 状態 (texture 含む) を FBO に描画して PNG 保存 */
        glViewport(0, 0, W, H);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        std::vector<mCutMesh*> imgMeshes = { screenMesh };
        std::vector<glm::vec4> imgColors = { glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
        draw_AllmCutMeshes(imgMeshes, *g_shaderProgram, *g_shaderProgramCube,
                           camPos, imgColors, modelMat, viewMat, projMat, 0);

        std::vector<unsigned char> pixels(W * H * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);   /* avoid row padding for non-mod-4 widths */
        glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        if (binarize) {
            /* 輝度 > 0 を 255、それ以外を 0 に (shader の lighting 影響を除去) */
            for (size_t i = 0; i < pixels.size(); i += 3) {
                int sum = (int)pixels[i] + (int)pixels[i+1] + (int)pixels[i+2];
                unsigned char v = (sum > 64 * 3) ? 255 : 0;  /* やや厳しめ閾値 */
                pixels[i] = pixels[i+1] = pixels[i+2] = v;
            }
        }

        int stride = W * 3;
        std::vector<unsigned char> flipped(pixels.size());
        for (int y = 0; y < H; y++)
            memcpy(&flipped[y * stride], &pixels[(H - 1 - y) * stride], stride);

        stbi_write_png(outPath.c_str(), W, H, 3, flipped.data(), stride);
        std::cout << "[TrajImage] Saved: " << outPath << std::endl;
    };

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {

        /* ---- Pass A: 通常の物理画像 (texture = 元画像) ---- */
        captureScreenMeshToPng(outDir + "/image.png", /*binarize=*/false);

        /* ---- Pass B: g_boundaryDistMap → 白黒 texture → 同カメラで撮影 ---- */
        if (g_boundaryDistMap.valid) {
            /* 2.1 g_boundaryDistMap.data (float, bd<9000 = mask 内) を
                   uint8 RGB の 3ch 白黒 texture buffer に変換。
                   mCutMesh::setRGBimage (line 114-118) と同様、stb_image の
                   top-left 原点を OpenGL の bottom-left 原点に合わせるため
                   upload 前に vertical flip する (これをやらないと liver_mask
                   が上下反転して出る)。 */
            const int mw = g_boundaryDistMap.width;
            const int mh = g_boundaryDistMap.height;
            std::vector<unsigned char> maskRGB(mw * mh * 3);
            for (int y = 0; y < mh; y++) {
                int srcY = mh - 1 - y;   /* 上下 flip */
                for (int x = 0; x < mw; x++) {
                    unsigned char v =
                        (g_boundaryDistMap.data[srcY * mw + x] < 9000.0f) ? 255 : 0;
                    int di = (y * mw + x) * 3;
                    maskRGB[di]   = v;
                    maskRGB[di+1] = v;
                    maskRGB[di+2] = v;
                }
            }

            /* 2.2 新規 GL texture にアップロード */
            GLuint maskTex = 0;
            glGenTextures(1, &maskTex);
            glBindTexture(GL_TEXTURE_2D, maskTex);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);   /* defensive */
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mw, mh, 0, GL_RGB,
                         GL_UNSIGNED_BYTE, maskRGB.data());
            /* binary mask なので NEAREST が望ましい (linear で中間値が出るのを防ぐ) */
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            /* 2.3 screenMesh の texture を一時的に差し替え */
            const GLuint origTexId = screenMesh->textureID;
            const bool   origHasTex = screenMesh->hasTexture;
            screenMesh->textureID = maskTex;
            screenMesh->hasTexture = true;

            /* 2.4 FBO をリバインドしてもう 1 回描画。binarize=true で
                   shader の lighting による中間値を完全除去 */
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            captureScreenMeshToPng(outDir + "/liver_mask.png", /*binarize=*/true);

            /* 2.5 復元 */
            screenMesh->textureID = origTexId;
            screenMesh->hasTexture = origHasTex;
            glDeleteTextures(1, &maskTex);
        } else {
            std::cerr << "[TrajImage] g_boundaryDistMap invalid, skipping liver_mask.png"
                      << " (SAM2 segmentation_mask.png not loaded?)" << std::endl;
        }

        glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    } else {
        std::cerr << "[TrajImage] FBO not complete, skipping image capture" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteTextures(1, &colorTex);
    glDeleteRenderbuffers(1, &depthRbo);
    glDeleteFramebuffers(1, &fbo);

    /* 3. depth slider を復元、mesh を再生成 */
    regenerateDepthMeshAuto(screenMesh, origDepthScale, gMeshScale);
    std::cout << "[TrajImage] depth slider restored to " << origDepthScale << std::endl;
}

static void poseAutoSaveBeforeRegistration() {
    g_poseLibrary.autoSaveLastRegistration(computeCurrentTransform());
}

// -------------------------------------------------------
// Unified comparison metric computation
// Same method for ALL entries: Liver visible vertices → depth front face
// Visible vertex INDICES are LOCKED at first call after registration.
// On subsequent calls (refine, manual save), the same indices are reused
// but current vertex POSITIONS are read — so the metric reflects the
// actual mesh geometry while keeping the source population identical.
// -------------------------------------------------------
/* CMA-ESループ中はtrue → computeUnifiedMetrics等の詳細ログを抑制 */
bool g_quietMetrics = false;

static void computeUnifiedMetrics() {
    Reg3DCustom::NoOpen3DRegistration reg;
    float zThresh = std::max(0.01f, gDepthScale * 0.05f);

    /* CMA-ESループ中はextractFrontFacePointsの出力も抑制 */
    std::streambuf* oldBuf = nullptr;
    std::ostringstream devNull;
    if (g_quietMetrics) {
        oldBuf = std::cout.rdbuf(devNull.rdbuf());
    }
    auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gGridWidth, gGridHeight(), zThresh);
    if (g_quietMetrics && oldBuf) {
        std::cout.rdbuf(oldBuf);
    }

    if (targetCloud->hasBoundaryDist()) {
        g_targetPoints.clear();
        g_cluster2Points.clear();
        for (size_t i = 0; i < targetCloud->size(); i++) {
            float bd = targetCloud->boundaryDist[i];
            if (bd >= 9000.0f) continue;
            if (bd < 12.0f)
                g_targetPoints.push_back(targetCloud->points[i]);
            else
                g_cluster2Points.push_back(targetCloud->points[i]);
        }
        if (!g_quietMetrics)
            std::cout << "[Boundary3D] boundary=" << g_targetPoints.size()
                      << " interior=" << g_cluster2Points.size() << std::endl;
    }

    auto sourceCloud = std::make_shared<Reg3DCustom::PointCloud>();
    const auto& verts = liverMesh3D->mVertices;
    for (size_t i = 0; i + 2 < verts.size(); i += 3) {
        sourceCloud->addPoint(glm::vec3(verts[i], verts[i + 1], verts[i + 2]));
    }

    Reg3DCustom::NanoflannAdaptor sourceAdaptor(sourceCloud->points);
    auto tree = Reg3DCustom::buildKDTree(sourceAdaptor);
    float max_dist_sq = 1.0f * 1.0f;

    std::vector<glm::vec3> src_pts, tgt_pts;
    float totalErr = 0.0f, sumSq = 0.0f, maxErr = 0.0f;
    for (size_t i = 0; i < targetCloud->size(); i++) {
        glm::vec3 tgtPt = targetCloud->points[i];
        size_t nnIdx; float dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, tgtPt, nnIdx, dist_sq)) {
            if (dist_sq < max_dist_sq) {
                float d = std::sqrt(dist_sq);
                tgt_pts.push_back(tgtPt);
                src_pts.push_back(sourceCloud->points[nnIdx]);
                totalErr += d;
                sumSq += d * d;
                if (d > maxErr) maxErr = d;
            }
        }
    }
    float n = tgt_pts.empty() ? 1.0f : (float)tgt_pts.size();

    registrationHandle.compRmse     = std::sqrt(sumSq / n);
    registrationHandle.compAvgError = totalErr / n;
    registrationHandle.compMaxError = maxErr;
    registrationHandle.compCount    = (int)tgt_pts.size();
    registrationHandle.compSource   = std::move(src_pts);
    registrationHandle.compTarget   = std::move(tgt_pts);

    if (!g_quietMetrics) {
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "[UnifiedMetrics T->S] Target: " << targetCloud->size()
                  << "  Matched: " << registrationHandle.compCount
                  << "  RMSE: " << registrationHandle.compRmse
                  << "  AvgErr: " << registrationHandle.compAvgError
                  << "  MaxErr: " << registrationHandle.compMaxError << std::endl;
    }

    /* ----- 3D Hausdorff: S->T 側 (両側 max を取って真の Hausdorff にする) -----
       既存の compMaxError は T->S 片側 max のみ。論文用に両側 max が必要。
       Source 集合: liver mesh の visible vertices (g_refineVertexIndices)。
                    可視性インデックスがない場合は全頂点で代用 (warning)。
       Target 集合: 既に上で取得した targetCloud (front face points)。
       T->S と異なり、S 側の各点について target の最近傍を検索するため
       targetCloud を KD-tree 化する。 */
    {
        Reg3DCustom::NanoflannAdaptor tgtAdaptor(targetCloud->points);
        auto tgtTree = Reg3DCustom::buildKDTree(tgtAdaptor);

        const auto& sv = liverMesh3D->mVertices;
        size_t nVerts = sv.size() / 3;
        bool useVisible = !g_refineVertexIndices.empty();
        const std::vector<size_t>& visIdx = g_refineVertexIndices;
        size_t nSource = useVisible ? visIdx.size() : nVerts;

        float maxS2T = 0.0f;
        size_t matchedS = 0;
        for (size_t k = 0; k < nSource; k++) {
            size_t i = useVisible ? visIdx[k] : k;
            if (i * 3 + 2 >= sv.size()) continue;
            glm::vec3 srcPt(sv[i*3], sv[i*3+1], sv[i*3+2]);
            size_t nnIdx; float dist_sq;
            if (Reg3DCustom::searchKNN1(*tgtTree, srcPt, nnIdx, dist_sq)) {
                if (dist_sq < max_dist_sq) {
                    float d = std::sqrt(dist_sq);
                    if (d > maxS2T) maxS2T = d;
                    matchedS++;
                }
            }
        }
        registrationHandle.compMaxS2T    = maxS2T;
        registrationHandle.compHausdorff = std::max(registrationHandle.compMaxError, maxS2T);

        if (!g_quietMetrics) {
            std::cout << "[Hausdorff3D] visible_src=" << nSource
                      << "  matched_S->T=" << matchedS
                      << "  MaxT2S=" << registrationHandle.compMaxError
                      << "  MaxS2T=" << maxS2T
                      << "  Hausdorff3D=" << registrationHandle.compHausdorff << std::endl;
        }
    }

    /* ----- 2D Hausdorff: シルエット境界の双方向最大距離 (画像 px 単位) -----
       g_boundaryDistMap が valid な時だけ計算 (depth マスクが必要)。
       view/projection はグローバルの現フレームを利用。Shift+E と完全同じ source
       シルエット (Fast ラスタライズ版) で計算するため、IoU 計算と同じ画素群が
       使われる -> Hausdorff と IoU の整合性が保たれる。 */
    if (g_boundaryDistMap.valid && liverMesh3D) {
        float h2d = 0.0f;
        int dInter, dUnion;
        double dMs;
        bool wasQuiet = g_quietMetrics;
        g_quietMetrics = true;  /* IoU の Sil2D-Fast ログを抑制 (二重出力防止) */
        CmaesRefine::computeSilhouette2DObjectiveFast(
            liverMesh3D, view, projection, /*step=*/8,
            &dInter, &dUnion, &dMs, &h2d);
        g_quietMetrics = wasQuiet;
        registrationHandle.sil2DHausdorff = h2d;
        registrationHandle.compIoU2D = (dUnion > 0) ? (float)dInter / (float)dUnion : 0.0f;

        if (!g_quietMetrics) {
            std::cout << "[Hausdorff2D] IoU=" << registrationHandle.compIoU2D
                      << "  H2D=" << std::fixed << std::setprecision(1)
                      << h2d << "px" << std::endl;
            std::cout << std::defaultfloat << std::setprecision(6);
        }
    } else {
        /* boundaryDistMap が未 build の時は IoU を 0 にしておく
           (poseSaveToLibrary の IoU 比較は currentIoU > 0 で守る) */
        registrationHandle.compIoU2D = 0.0f;
    }
}

static void startNewSession() {
    g_sessionId++;
    g_sessionBipopN      = 0;
    g_sessionRefineN     = 0;
    g_sessionSilhouetteN = 0;
    g_bestSessionCompRmse = FLT_MAX;
    g_bestSessionIoU2D    = 0.0f;
    g_bestSessionVertices.clear();
    g_bestSessionNormals.clear();
    g_currentOrientRunCount = 0;
    g_stepStartTime = std::chrono::steady_clock::now();
    std::cout << "[Session] New session #" << g_sessionId << std::endl;
}

/* ============================================================
 *  Shift+P protocol helpers (Step 4a)
 * ------------------------------------------------------------
 *  These are pure additions — nothing here is called unless the
 *  Shift+P protocol is active, so existing UI behavior is
 *  completely unchanged.
 * ============================================================ */

/* Build a perturbation matrix in the same spirit as
 * getPresetRotation(): rotate around the mesh centroid, then
 * apply a small world-space translation. Used by Condition B of
 * the protocol to create operator-pose variations around TOP.
 *
 *   rxDeg, ryDeg, rzDeg : rotation about X/Y/Z axes (degrees)
 *   tx, ty, tz          : translation in scene units
 *                         (±0.1 scene-unit recommended to stay
 *                          safely inside CMA-ES's ±1 basin)
 *   centroid            : liver mesh centroid in world coords
 *
 * Order of multiplication mirrors getPresetRotation():
 *   T_world * (ToOrigin → Rz → Ry → Rx → FromOrigin)
 */
static glm::mat4 buildPerturbation(float rxDeg, float ryDeg, float rzDeg,
                                   float tx, float ty, float tz,
                                   const glm::vec3& centroid)
{
    glm::mat4 toOrigin   = glm::translate(glm::mat4(1.0f), -centroid);
    glm::mat4 Rx         = glm::rotate(glm::mat4(1.0f), glm::radians(rxDeg), glm::vec3(1,0,0));
    glm::mat4 Ry         = glm::rotate(glm::mat4(1.0f), glm::radians(ryDeg), glm::vec3(0,1,0));
    glm::mat4 Rz         = glm::rotate(glm::mat4(1.0f), glm::radians(rzDeg), glm::vec3(0,0,1));
    glm::mat4 fromOrigin = glm::translate(glm::mat4(1.0f), centroid);
    glm::mat4 T          = glm::translate(glm::mat4(1.0f), glm::vec3(tx, ty, tz));
    return T * fromOrigin * Rz * Ry * Rx * toOrigin;
}

/* Synchronous Refine runner.
 *
 * The UI's a.onRefine is a toggle (press-start / press-stop) and
 * steps are advanced by the main render loop's REFINING block.
 * For the protocol we need a same-thread, blocking execution that
 * returns a complete before/after report. This function replicates
 * the three phases (init / step loop / finalize) inline so the UI
 * path is untouched.
 *
 *   max_iter : hard cap on refine iterations
 *
 * Return fields:
 *   initialRmse      : compRMSE measured right after init
 *   finalRmse        : compRMSE after this function returns
 *                      (= initialRmse if nothing improved)
 *   improved         : true iff finalRmse < initialRmse
 *                      (mesh is kept at best state when true,
 *                       restored to initial state when false)
 *   iterationsRun    : actual number of refineStep calls executed
 *   bestIteration    : iteration index that produced the best RMSE
 *   elapsedSec       : wall-clock time for init+loop+finalize
 *
 * Early-stop rules (match the UI REFINING block):
 *   - correspondenceCount < 6 → converged
 *   - step.converged         → converged
 *   - worseCount >= 30       → early stop
 *
 * Note: computeUnifiedMetrics() is called once per accepted step,
 * same as the UI path — so per-iteration cost is identical.
 */
struct RefineSyncResult {
    float initialRmse    = 0.0f;
    float finalRmse      = 0.0f;
    bool  improved       = false;
    int   iterationsRun  = 0;
    int   bestIteration  = 0;
    float elapsedSec     = 0.0f;
};

static RefineSyncResult runRefineSync(int max_iter = 60)
{
    RefineSyncResult r;
    auto t0 = std::chrono::steady_clock::now();

    /* Guard: require a completed registration and visibility indices */
    if (registrationHandle.state != RegistrationData::REGISTERED) {
        std::cerr << "[RefineSync] Not REGISTERED; skipping." << std::endl;
        return r;
    }
    if (g_refineVertexIndices.empty()) {
        std::cerr << "[RefineSync] No visible vertex indices; skipping." << std::endl;
        return r;
    }

    /* --- init phase (same as a.onRefine REGISTERED branch) --- */
    auto organs = getOrganList();
    NormalRefine::RefineParams params;
    params.useZWeight      = true;
    params.zWeightBoundary = 0.05f;
    params.zWeightInterior = 0.30f;
    params.boundaryWidth   = 8.0f;
    params.boundaryBoost   = 3.0f;

    if (!NormalRefine::initRefine(g_refineState, liverMesh3D,
                                  g_refineVertexIndices,
                                  screenMesh, organs,
                                  gGridWidth, gGridHeight(), gDepthScale, params,
                                  NormalRefine::NORMAL_COMPAT)) {
        std::cerr << "[RefineSync] Initialization failed" << std::endl;
        computeUnifiedMetrics();
        r.initialRmse = r.finalRmse = registrationHandle.compRmse;
        r.elapsedSec  = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - t0).count();
        return r;
    }

    computeUnifiedMetrics();
    g_refineState.initialRMSE = registrationHandle.compRmse;
    g_refineState.bestRMSE    = registrationHandle.compRmse;
    r.initialRmse             = registrationHandle.compRmse;
    registrationHandle.state  = RegistrationData::REFINING;

    /* --- step loop (same logic as main render-loop REFINING branch) --- */
    for (int iter = 0; iter < max_iter; iter++) {
        if (!g_refineState.active) break;

        auto step = NormalRefine::refineStep(g_refineState, OrbitCam.cameraDirection);

        if (step.correspondenceCount >= 6 && !step.converged) {
            NormalRefine::applyIncrementalTransform(step.incrementalTransform,
                                                    g_refineState.organMeshes);
            g_refineState.cumulativeTransform =
                step.incrementalTransform * g_refineState.cumulativeTransform;

            computeUnifiedMetrics();
            float rmse = registrationHandle.compRmse;
            if (rmse < g_refineState.bestRMSE) {
                g_refineState.bestRMSE                = rmse;
                g_refineState.bestCumulativeTransform = g_refineState.cumulativeTransform;
                g_refineState.bestIteration           = g_refineState.totalIterations;
                g_refineState.worseCount              = 0;
            } else {
                g_refineState.worseCount++;
            }
            if (g_refineState.worseCount >= 30) {
                step.converged = true;
                std::cout << "[RefineSync] Early stop: RMSE worsening for 30 iters" << std::endl;
            }
        }

        r.iterationsRun = iter + 1;
        if (step.converged) break;
    }

    /* --- finalize (same as a.onRefine REFINING branch) --- */
    g_refineState.active     = false;
    registrationHandle.state = RegistrationData::REGISTERED;
    r.improved               = g_refineState.bestRMSE < g_refineState.initialRMSE;
    r.bestIteration          = g_refineState.bestIteration;

    g_refineState.restoreMeshes();
    if (r.improved) {
        NormalRefine::applyIncrementalTransform(
            g_refineState.bestCumulativeTransform,
            g_refineState.organMeshes);
    }
    computeUnifiedMetrics();
    r.finalRmse = registrationHandle.compRmse;

    /* Mirror the UI path's bookkeeping so downstream code sees
     * consistent state (matches the UI REFINING stop branch). */
    registrationHandle.refineCount++;
    registrationHandle.refineInitialRMSE   = g_refineState.initialRMSE;
    registrationHandle.refineBestRMSE      = g_refineState.bestRMSE;
    registrationHandle.refineBestIteration = g_refineState.bestIteration;

    r.elapsedSec = std::chrono::duration<float>(
                       std::chrono::steady_clock::now() - t0).count();

    std::cout << "[RefineSync] initial=" << r.initialRmse
              << " final=" << r.finalRmse
              << (r.improved ? " [IMPROVED]" : " [reverted]")
              << "  iters=" << r.iterationsRun
              << " bestIter=" << r.bestIteration
              << "  time=" << r.elapsedSec << "s" << std::endl;
    return r;
}

/* ============================================================
 *  Shift+P MICCAI-WS protocol runner (Step 4, simplified)
 * ------------------------------------------------------------
 *  Simple, synchronous implementation — no state machine, no
 *  ImGui overlay, no frame-based advance. Blocks the UI for
 *  ~50 seconds per image while running:
 *
 *    for each of 10 trials:
 *        restore mesh → apply TOP preset → apply perturbation (B only)
 *        setFgrSeed(trial_seed)
 *        HemiAuto × 2       (each continues from previous pose)
 *        CMA-ES (BIPOP) × 10 (each continues from previous pose;
 *                              each call contains 10 internal restarts)
 *        Refine × 1
 *        log everything
 *
 *  10 trials: B0..B4 (pose perturbation, seed fixed at 42)
 *           + C0..C4 (TOP pose, seed ∈ {42,123,456,789,2024})
 *  B0 and C0 are complete duplicates — serves as internal
 *  determinism integrity check.
 *
 *  Called from the Shift+P key handler (REGISTRATION_MODE only).
 * ============================================================ */

struct ProtocolTrial {
    char        condition;   /* 'B' or 'C' */
    int         trial_idx;   /* 0..4 */
    float       rx_deg, ry_deg, rz_deg;   /* pose perturbation (B only) */
    float       tx, ty, tz;               /* translation perturb (B only) */
    unsigned    trial_seed;
    std::string label;
};

static std::vector<ProtocolTrial> buildProtocolTrialTable() {
    const unsigned BSEED = 42u;
    std::vector<ProtocolTrial> T;
    T.reserve(10);

    /* Condition B: pose perturbation, seed fixed at 42.
     * Values tuned to simulate realistic operator placement error
     * while staying within HemiAuto (FGR+ICP) capture radius:
     *   rotation ±10°
     *   translation ±0.5 (= 5% of screen mesh width, which is 10 scene units)
     * The Shift+B debug toggle (see glfw_onKey) lets the user preview
     * these initial poses interactively before running the protocol. */
    T.push_back({ 'B', 0,  0.0f, 0.0f,  0.0f,   0.0f, 0.0f,  0.0f,   BSEED, "TOP"            });
    T.push_back({ 'B', 1,+10.0f, 0.0f,  0.0f,  +0.5f, 0.0f,  0.0f,   BSEED, "Rx+10_tx+0.5"   });
    T.push_back({ 'B', 2,-10.0f, 0.0f,  0.0f,  -0.5f, 0.0f,  0.0f,   BSEED, "Rx-10_tx-0.5"   });
    T.push_back({ 'B', 3,  0.0f,+10.0f, 0.0f,   0.0f,+0.5f,  0.0f,   BSEED, "Ry+10_ty+0.5"   });
    T.push_back({ 'B', 4,  0.0f, 0.0f,+10.0f,   0.0f, 0.0f, +0.5f,   BSEED, "Rz+10_tz+0.5"   });

    /* Condition C: pose fixed at TOP, seed varies */
    const unsigned CSEEDS[5] = { 42u, 123u, 456u, 789u, 2024u };
    for (int i = 0; i < 5; i++) {
        T.push_back({ 'C', i, 0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f,
                      CSEEDS[i], std::string("seed=") + std::to_string(CSEEDS[i]) });
    }
    return T;
}

/* ------------------------------------------------------------
 *  Shift+B debug toggle: preview B0..B4 initial poses.
 * ------------------------------------------------------------
 *  Cycles the mesh through the starting poses that the protocol
 *  would place it in at the beginning of each B-trial (before any
 *  HemiAuto / CMA-ES / Refine runs). Useful for visually inspecting:
 *    - whether the ±10° / ±0.2 perturbation magnitudes are sensible
 *    - which trial the failed runs (B2 in image1) originated from
 *    - recreating a specific trial's initial state so the user can
 *      manually click Hemi Auto / CMA-ES / Refine to reproduce it
 *
 *  State persists across calls (module-level counter). Works any
 *  time the mesh is loaded — no protocol run required.
 * ------------------------------------------------------------ */

static int g_shiftBToggleIdx = -1;   /* -1 = not yet toggled; 0..4 = currently showing Bn */

static void showTrialInitPose(int b_idx) {
    if (b_idx < 0 || b_idx > 4) return;
    if (!liverMesh3D || g_initOrganVertices.empty()) {
        std::cerr << "[Shift+B] Mesh or initial vertices not loaded." << std::endl;
        return;
    }

    auto trials = buildProtocolTrialTable();
    if (b_idx >= (int)trials.size() || trials[b_idx].condition != 'B') {
        std::cerr << "[Shift+B] B" << b_idx << " not in trial table." << std::endl;
        return;
    }
    const auto& T = trials[b_idx];

    auto organs = getOrganList();
    if (g_initOrganVertices.size() != organs.size()) {
        std::cerr << "[Shift+B] organ count mismatch." << std::endl;
        return;
    }

    /* (1) Restore mesh to initial state */
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            organs[i]->mVertices = g_initOrganVertices[i];
            organs[i]->mNormals  = g_initOrganNormals[i];
            setUp(*organs[i]);
        }
    }

    /* (2) Apply TOP preset rotation */
    {
        glm::vec3 centroid = computeMeshCentroidFromVertices(liverMesh3D->mVertices);
        glm::mat4 Rtop = getPresetRotation(RegistrationData::PRESET_TOP, centroid);
        for (auto* m : organs) {
            if (m) { applyMatrixToMeshVerticesAndNormals(m, Rtop); setUp(*m); }
        }
    }

    /* (3) Apply B-perturbation (identity for B0) */
    if (T.rx_deg != 0 || T.ry_deg != 0 || T.rz_deg != 0 ||
        T.tx    != 0 || T.ty    != 0 || T.tz    != 0) {
        glm::vec3 centroid = computeMeshCentroidFromVertices(liverMesh3D->mVertices);
        glm::mat4 P = buildPerturbation(T.rx_deg, T.ry_deg, T.rz_deg,
                                        T.tx, T.ty, T.tz, centroid);
        for (auto* m : organs) {
            if (m) { applyMatrixToMeshVerticesAndNormals(m, P); setUp(*m); }
        }
    }

    /* (4) Reset registration state so UI doesn't show stale metrics.
     * The user can now manually click Hemi Auto / CMA-ES / Refine to
     * reproduce this trial interactively. */
    resetRegistrationState();

    std::cout << "[Shift+B] Now showing B" << b_idx
              << " initial pose (" << T.label << ")"
              << "  — press Hemi Auto to reproduce the protocol from here."
              << std::endl;
}

/* ------------------------------------------------------------
 * Synchronous replica of the a.onHemiAuto lambda body
 * (main.cpp ~line 1353) with UI side-effects removed:
 *   - no poseAutoSaveBeforeRegistration
 *   - no gUIManager.state.regMethod = 1
 *   - no poseSaveToLibrary (suppressed anyway via g_suppressPoseLibSave)
 *   - no computeIdealVoxelSizes (UI telemetry only)
 *
 * Determinism: caller must have invoked Reg3DCustom::setFgrSeed(seed)
 * before calling this function.
 * Returns compRMSE, or -1 on failure (too few visible pts).
 * ------------------------------------------------------------ */
static float runProtocolHemiAutoStep() {
    resetRegistrationState();

    Reg3D::BVHTree bvh;
    bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
    auto vis = Reg3DCustom::extractVisibleVerticesCustom(
        *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);
    if (vis.cloud->size() < 50) {
        std::cerr << "[Protocol/HemiAuto] too few visible pts ("
                  << vis.cloud->size() << ")" << std::endl;
        return -1.0f;
    }

    g_cluster1Points = vis.points;
    g_cluster2Points.clear();
    g_refineVertexIndices = vis.vertexIndices;

    auto organs = getOrganList();
    Reg3DCustom::performRegistrationSingleMesh(
        organs, liverMesh3D, vis.vertexIndices,
        screenMesh, OrbitCam.cameraPos,
        gGridWidth, gGridHeight(),
        15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);

    computeUnifiedMetrics();
    return registrationHandle.compRmse;
}

/* ------------------------------------------------------------
 * Synchronous replica of the a.onBipopCmaes lambda body
 * (main.cpp ~line 1478) with UI side-effects removed:
 *   - no poseAutoSaveBeforeRegistration
 *   - no g_sessionBipopN++, gUIManager.state.regMethod = 3
 *   - no poseSaveToLibrary
 *
 * Keeps the exact 10-restart BIPOP loop structure of the UI
 * button, including Regime1/Regime2 alternation, sigma0 and
 * perturbation generation from an mt19937 rng, start_v/best_v
 * tracking, and final best restoration.
 *
 * Two determinism hooks relative to the UI version:
 *   - outer rng is seeded with (trial_seed + 1000) instead of
 *     std::random_device{}() (makes Regime/perturbation generation
 *     reproducible)
 *   - CmaesRefine::Params::rng_seed is set to (trial_seed + 2000
 *     + call_idx * 10 + run) — unique per restart, reproducible
 *
 * Returns the compRMSE of the best-of-10 restart for this call.
 * ------------------------------------------------------------ */
static float runProtocolCmaesStep(unsigned trial_seed, int call_idx) {
    auto organs = getOrganList();
    computeUnifiedMetrics();
    float rmse_before = registrationHandle.compRmse;

    /* Snapshot current pose as the "start" for all 10 restarts
     * (exactly mirrors UI behavior at line 1494-1498). */
    std::vector<std::vector<GLfloat>> start_v(organs.size());
    std::vector<std::vector<GLfloat>> start_n(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            start_v[i] = organs[i]->mVertices;
            start_n[i] = organs[i]->mNormals;
        }
    }

    float best_rmse = rmse_before;
    std::vector<std::vector<GLfloat>> best_v = start_v;
    std::vector<std::vector<GLfloat>> best_n = start_n;

    const int N_STARTS = 10;
    /* Determinism: seed the outer rng from trial_seed+1000 rather than
     * std::random_device{}() as the UI does. Same draw sequence every
     * time this (trial_seed, call_idx) pair runs. */
    std::mt19937 rng(trial_seed + 1000u + (unsigned)call_idx * 97u);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int run = 0; run < N_STARTS; run++) {
        /* Reset to start pose (UI line 1509-1511) */
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) {
                organs[i]->mVertices = start_v[i];
                organs[i]->mNormals  = start_n[i];
                setUp(*organs[i]);
            }
        }

        CmaesRefine::Params p;
        p.verbose        = true;
        p.log_every      = 100;
        p.save_debug_jpg = false;
        /* Determinism: uniquely seed each restart */
        p.rng_seed       = trial_seed + 2000u + (unsigned)call_idx * 10u + (unsigned)run;

        float tx_perturb = 0, ty_perturb = 0, tz_perturb = 0;
        float rx_perturb = 0, ry_perturb = 0, rz_perturb = 0;
        float sc_perturb = 1.0f;
        std::string regime;

        /* Regime alternation — identical to UI */
        if (run % 2 == 0) {
            p.sigma0   = 0.3 + dist01(rng) * 0.4;
            tx_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
            ty_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
            tz_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
            rx_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
            ry_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
            rz_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
            sc_perturb = 0.95f + dist01(rng) * 0.10f;
            regime = "Regime2(local)";
        } else {
            p.sigma0   = 0.5 + dist01(rng) * 0.5;
            tx_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
            ty_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
            tz_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
            rx_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
            ry_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
            rz_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
            sc_perturb = 0.90f + dist01(rng) * 0.20f;
            regime = "Regime1(global)";
        }

        if (run > 0) {
            CmaesRefine::applyIncrementalSRT(organs,
                                             tx_perturb, ty_perturb, tz_perturb,
                                             rx_perturb, ry_perturb, rz_perturb,
                                             sc_perturb);
            for (size_t i = 0; i < organs.size(); i++)
                if (organs[i]) setUp(*organs[i]);
        }

        std::cout << "[Protocol/CMA-ES call " << call_idx << "] Run " << (run+1)
                  << "/" << N_STARTS << "  " << regime
                  << "  sigma0=" << std::fixed << std::setprecision(2)
                  << p.sigma0 << std::endl;

        CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                                 gGridWidth, gGridHeight(),
                                                 gDepthScale, p);
        computeUnifiedMetrics();
        float rmse_run = registrationHandle.compRmse;
        std::cout << "[Protocol/CMA-ES call " << call_idx << "] Run " << (run+1)
                  << " compRMSE=" << std::setprecision(6) << rmse_run
                  << (r.improved ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;

        if (rmse_run < best_rmse) {
            best_rmse = rmse_run;
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i]) {
                    best_v[i] = organs[i]->mVertices;
                    best_n[i] = organs[i]->mNormals;
                }
            }
        }
    }

    /* Restore to best (UI line 1569-1571) */
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            organs[i]->mVertices = best_v[i];
            organs[i]->mNormals  = best_n[i];
            setUp(*organs[i]);
        }
    }
    computeUnifiedMetrics();
    std::cout << "[Protocol/CMA-ES call " << call_idx << "] Best: "
              << rmse_before << " -> " << best_rmse
              << (best_rmse < rmse_before ? " [IMPROVED]" : " [NO CHANGE]")
              << std::endl;
    return best_rmse;
}

/* ------------------------------------------------------------
 * Populate the "metrics" subset of a ProtocolEntry from the
 * current registrationHandle state. Caller fills identity
 * (image, cond, trial_idx, phase, call_idx, seeds, perturbation)
 * and timing separately.
 * ------------------------------------------------------------ */
static void fillProtocolEntryMetrics(ProtocolEntry& e) {
    e.comp_rmse      = registrationHandle.compRmse;
    e.comp_avg_error = registrationHandle.compAvgError;
    e.comp_max_error = registrationHandle.compMaxError;
    e.comp_count     = registrationHandle.compCount;
    e.base_fitness   = registrationHandle.fitness;
    e.base_icp_rmse  = registrationHandle.icpRmse;
    e.base_scale     = registrationHandle.scaleFactor;

    /* Pose transform: reuse the same Procrustes-based estimation
     * that PoseLibrary uses for its interactive entries, so the
     * CSV's m00-m33 columns can be applied back to the initial
     * pose via PoseLibrary::applyTransformToMeshes() at any time.
     * This is what makes the "restore best after protocol" feature
     * possible without having to re-run any trial. */
    if (!g_initOrganVertices.empty() && liverMesh3D &&
        !liverMesh3D->mVertices.empty() &&
        g_initOrganVertices[0].size() == liverMesh3D->mVertices.size()) {
        e.transform = PoseLibrary::computeTransformFromLiver(
            g_initOrganVertices[0],      /* initial pose */
            liverMesh3D->mVertices);     /* current pose */
    } else {
        e.transform = glm::mat4(1.0f);
    }
}

/* ------------------------------------------------------------
 * Main protocol runner.
 * ------------------------------------------------------------ */
static void runProtocolBatch(const std::string& imageName) {
    if (currentMainMode != REGISTRATION_MODE) {
        std::cerr << "[Protocol] Not in REGISTRATION_MODE; aborting." << std::endl;
        return;
    }
    if (!liverMesh3D || !screenMesh) {
        std::cerr << "[Protocol] liverMesh3D or screenMesh is null; aborting." << std::endl;
        return;
    }
    if (g_initOrganVertices.empty()) {
        std::cerr << "[Protocol] g_initOrganVertices is empty; load an image first." << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Shift+P Protocol Run  image=" << imageName << std::endl;
    std::cout << "========================================" << std::endl;

    auto trials = buildProtocolTrialTable();

    g_protocolLog.begin(imageName, "./");
    g_suppressPoseLibSave = true;
    auto batch_t0 = std::chrono::steady_clock::now();

    auto organs = getOrganList();
    if (g_initOrganVertices.size() != organs.size()) {
        std::cerr << "[Protocol] organ count mismatch ("
                  << g_initOrganVertices.size() << " vs " << organs.size()
                  << "); aborting." << std::endl;
        g_suppressPoseLibSave = false;
        return;
    }

    for (size_t ti = 0; ti < trials.size(); ti++) {
        const auto& T = trials[ti];
        std::cout << "\n----- Trial " << (ti + 1) << "/" << trials.size()
                  << "  " << T.condition << T.trial_idx
                  << " (" << T.label << ", seed=" << T.trial_seed << ") -----"
                  << std::endl;

        /* (1) Restore mesh to initial state. */
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) {
                organs[i]->mVertices = g_initOrganVertices[i];
                organs[i]->mNormals  = g_initOrganNormals[i];
                setUp(*organs[i]);
            }
        }

        /* (2) Apply TOP preset rotation (matches UI preset TOP). */
        {
            glm::vec3 centroid = computeMeshCentroidFromVertices(liverMesh3D->mVertices);
            glm::mat4 Rtop = getPresetRotation(RegistrationData::PRESET_TOP, centroid);
            for (auto* m : organs) {
                if (m) {
                    applyMatrixToMeshVerticesAndNormals(m, Rtop);
                    setUp(*m);
                }
            }
        }

        /* (3) For Condition B, apply the pose perturbation around centroid. */
        if (T.condition == 'B' &&
            (T.rx_deg != 0 || T.ry_deg != 0 || T.rz_deg != 0 ||
             T.tx    != 0 || T.ty    != 0 || T.tz    != 0)) {
            glm::vec3 centroid = computeMeshCentroidFromVertices(liverMesh3D->mVertices);
            glm::mat4 P = buildPerturbation(T.rx_deg, T.ry_deg, T.rz_deg,
                                            T.tx, T.ty, T.tz, centroid);
            for (auto* m : organs) {
                if (m) {
                    applyMatrixToMeshVerticesAndNormals(m, P);
                    setUp(*m);
                }
            }
        }

        /* (4) Set FGR seed for this trial. All HemiAuto calls in this
         * trial use the same seed — the 2 calls are chained via pose
         * continuation, not seed variation. */
        Reg3DCustom::setFgrSeed(T.trial_seed);

        /* (5) HemiAuto × 2 (chained, pose continues) */
        for (int h = 0; h < 2; h++) {
            auto t0 = std::chrono::steady_clock::now();
            float rmse = runProtocolHemiAutoStep();
            float dt = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - t0).count();
            (void)rmse;  /* rmse also captured via computeUnifiedMetrics → metrics */

            ProtocolEntry e;
            e.image_name        = imageName;
            e.condition         = T.condition;
            e.trial_idx         = T.trial_idx;
            e.perturbation      = T.label;
            e.trial_seed        = T.trial_seed;
            e.fgr_seed          = T.trial_seed;
            e.bipop_outer_seed  = T.trial_seed + 1000u;
            e.cma_inner_seed    = T.trial_seed + 2000u;
            e.phase             = 'H';
            e.call_idx          = h;
            fillProtocolEntryMetrics(e);
            e.elapsed_sec       = dt;
            g_protocolLog.add(e);
        }

        /* (6) CMA-ES × 10 (each call chains from previous; each call
         * contains 10 internal restarts as per UI BIPOP behavior) */
        for (int c = 0; c < 10; c++) {
            auto t0 = std::chrono::steady_clock::now();
            float rmse = runProtocolCmaesStep(T.trial_seed, c);
            float dt = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - t0).count();
            (void)rmse;

            ProtocolEntry e;
            e.image_name        = imageName;
            e.condition         = T.condition;
            e.trial_idx         = T.trial_idx;
            e.perturbation      = T.label;
            e.trial_seed        = T.trial_seed;
            e.fgr_seed          = T.trial_seed;
            e.bipop_outer_seed  = T.trial_seed + 1000u + (unsigned)c * 97u;
            e.cma_inner_seed    = T.trial_seed + 2000u + (unsigned)c * 10u;
            e.phase             = 'C';
            e.call_idx          = c;
            fillProtocolEntryMetrics(e);
            e.elapsed_sec       = dt;
            g_protocolLog.add(e);
        }

        /* (7) Refine × 1 */
        {
            /* capture rmse right before Refine for refine_delta_rmse */
            computeUnifiedMetrics();
            float rmse_before_refine = registrationHandle.compRmse;

            auto t0 = std::chrono::steady_clock::now();
            RefineSyncResult rr = runRefineSync();
            float dt = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - t0).count();

            ProtocolEntry e;
            e.image_name        = imageName;
            e.condition         = T.condition;
            e.trial_idx         = T.trial_idx;
            e.perturbation      = T.label;
            e.trial_seed        = T.trial_seed;
            e.fgr_seed          = T.trial_seed;
            e.bipop_outer_seed  = T.trial_seed + 1000u;
            e.cma_inner_seed    = T.trial_seed + 2000u;
            e.phase             = 'R';
            e.call_idx          = 0;
            fillProtocolEntryMetrics(e);
            e.refine_applied    = 1;
            e.refine_improved   = rr.improved ? 1 : 0;
            e.refine_delta_rmse = rr.finalRmse - rmse_before_refine;
            e.elapsed_sec       = dt;
            g_protocolLog.add(e);
        }
    }

    /* Restore state */
    Reg3DCustom::setFgrSeed(0);          /* return to non-deterministic default */
    g_suppressPoseLibSave = false;

    float total = std::chrono::duration<float>(
                     std::chrono::steady_clock::now() - batch_t0).count();

    /* ------------------------------------------------------------
     * Final step: restore mesh to the best-compRMSE entry so the
     * UI displays the best result for screenshot capture. We use
     * the same PoseLibrary::applyTransformToMeshes() path as the
     * interactive "Apply" button, feeding it the best entry's
     * transform (already computed by fillProtocolEntryMetrics and
     * stored in the CSV's m00-m33 columns).
     *
     * Deterministic: the transform was computed via Procrustes from
     * the entry's mesh state and stored before we moved on, so
     * applying it to g_initOrganVertices reproduces the exact best
     * pose up to Procrustes inversion error (<< pixel level).
     * ------------------------------------------------------------ */
    const auto& entries = g_protocolLog.entries;
    if (!entries.empty()) {
        size_t best_i = 0;
        float  best_rmse = entries[0].comp_rmse;
        for (size_t i = 1; i < entries.size(); i++) {
            if (entries[i].comp_rmse < best_rmse) {
                best_rmse = entries[i].comp_rmse;
                best_i    = i;
            }
        }
        const auto& be = entries[best_i];
        std::cout << "[Protocol] Best entry: #" << be.entry_id
                  << "  " << be.condition << be.trial_idx
                  << "  phase=" << be.phase << " call=" << be.call_idx
                  << "  compRMSE=" << be.comp_rmse
                  << "  (" << be.perturbation << ")" << std::endl;

        /* Apply best transform to the initial mesh vertices.
         * This is identical to what poseApplyEntry() does for UI. */
        PoseLibrary::applyTransformToMeshes(
            be.transform, g_initOrganVertices, g_initOrganNormals, organs);
        for (auto* m : organs) if (m) setUp(*m);

        /* Sync registrationHandle so the UI sees the best metrics. */
        computeUnifiedMetrics();
        registrationHandle.state            = RegistrationData::REGISTERED;
        registrationHandle.useRegistration  = true;
        registrationHandle.fitness          = be.base_fitness;
        registrationHandle.icpRmse          = be.base_icp_rmse;
        registrationHandle.scaleFactor      = be.base_scale;
        std::cout << "[Protocol] Mesh restored to best pose (compRMSE="
                  << registrationHandle.compRmse << ")" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Protocol complete:  " << total << " s" << std::endl;
    std::cout << "  Entries logged: " << g_protocolLog.size() << std::endl;
    std::cout << "========================================" << std::endl;

    g_protocolLog.save();
}

static void poseSaveToLibrary(SaveCriterion crit = SaveCriterion::RMSE) {
    /* Step 4a: Shift+P protocol runs bypass the interactive pose library
     * to avoid polluting it with ~130 automated entries. The UI button
     * and key bindings always operate with g_suppressPoseLibSave=false. */
    if (g_suppressPoseLibSave) return;

    if (registrationHandle.state != RegistrationData::REGISTERED) {
        std::cout << "[PoseLibrary] No registration to save" << std::endl;
        return;
    }

    float elapsedSec = std::chrono::duration<float>(
                           std::chrono::steady_clock::now() - g_stepStartTime).count();

    float currentRmse = registrationHandle.compRmse;
    float currentIoU  = registrationHandle.compIoU2D;
    auto organs = getOrganList();

    PoseEntry::Method method;
    int rm = gUIManager.state.regMethod;
    if      (rm == 0) method = PoseEntry::FULL_AUTO;
    else if (rm == 1) method = PoseEntry::HEMI_AUTO;
    else if (rm == 2) method = PoseEntry::UMEYAMA;
    else if (rm == 3) method = PoseEntry::BIPOP_CMAES;
    else if (rm == 5) method = PoseEntry::SILHOUETTE_ALIGN;
    else              method = PoseEntry::HEMI_AUTO;

    if (registrationHandle.refineCount > 0 &&
        registrationHandle.refineCount > g_sessionRefineN) {
        method = PoseEntry::REFINE;
        g_sessionRefineN = registrationHandle.refineCount;
    }

    /* Multi-criterion accept logic.
       - rmseImproved : compRmse <= session-best (smaller is better)
       - iouImproved  : compIoU2D > session-best + eps (larger is better, eps=1e-4)
       - accept :
         * RMSE   -> rmseImproved (legacy: HemiAuto / BIPOP / Refine / Umeyama)
         * IOU    -> iouImproved && currentIoU > 0 (Shift+E; guard against IoU=0
                    sentinel that means boundaryDistMap not built yet)
         * EITHER -> rmseImproved || iouImproved
       g_bestSessionVertices semantics : *the most recent accepted state*,
       NOT the RMSE-best state. This change is required so that an IoU-driven
       accept (RMSE may have degraded) is not undone by a subsequent RMSE-driven
       reject. */
    bool rmseImproved = (currentRmse <= g_bestSessionCompRmse);
    bool iouImproved  = (currentIoU  > g_bestSessionIoU2D + 1e-4f) && (currentIoU > 0.0f);
    bool accept = false;
    switch (crit) {
    case SaveCriterion::RMSE:   accept = rmseImproved; break;
    case SaveCriterion::IOU:    accept = iouImproved;  break;
    case SaveCriterion::EITHER: accept = rmseImproved || iouImproved; break;
    }

    const char* critName = (crit == SaveCriterion::RMSE)   ? "RMSE"
                         : (crit == SaveCriterion::IOU)    ? "IOU"
                                                           : "EITHER";

    if (accept) {
        g_currentOrientRunCount++;

        /* On accept, ALWAYS update both reference values to current state.
           The "g_bestSession*" trackers now mean "last-accepted reference",
           NOT "best ever seen in session". This matches the snapshot vertices
           semantics (also updated on every accept).

           Why this matters:
             Without this, e.g. Shift+E accept (IoU improved, RMSE worsened
             from 0.18 to 0.22) leaves g_bestSessionCompRmse=0.18. A subsequent
             Shift+V whose RMSE improves over the current 0.22 (say to 0.20)
             would be falsely rejected because 0.20 > 0.18.
             With the fix: g_bestSessionCompRmse becomes 0.22 after Shift+E,
             so Shift+V's 0.20 satisfies 0.20 <= 0.22 -> accept.

           IoU=0 sentinel guard: don't overwrite a known-good IoU reference
           with a 0 value (which means boundaryDistMap not built). */
        float prevRmseRef = g_bestSessionCompRmse;
        float prevIouRef  = g_bestSessionIoU2D;
        g_bestSessionCompRmse = currentRmse;
        if (currentIoU > 0.0f) g_bestSessionIoU2D = currentIoU;

        if (currentRmse < prevRmseRef)
            std::cout << "[Session] CompRMSE reference: " << prevRmseRef
                      << " -> " << currentRmse << " [improved]" << std::endl;
        else if (currentRmse > prevRmseRef)
            std::cout << "[Session] CompRMSE reference: " << prevRmseRef
                      << " -> " << currentRmse << " [regressed but accepted via "
                      << critName << "]" << std::endl;
        if (currentIoU > 0.0f) {
            if (currentIoU > prevIouRef + 1e-4f)
                std::cout << "[Session] IoU2D reference: " << prevIouRef
                          << " -> " << currentIoU << " [improved]" << std::endl;
            else if (currentIoU < prevIouRef - 1e-4f)
                std::cout << "[Session] IoU2D reference: " << prevIouRef
                          << " -> " << currentIoU << " [regressed but accepted via "
                          << critName << "]" << std::endl;
        }

        /* Snapshot current state as the "last accepted" for revert anchor.
           Always overwrite (not just on RMSE-best) so an IoU-accept is preserved. */
        g_bestSessionVertices.resize(organs.size());
        g_bestSessionNormals.resize(organs.size());
        for (size_t i = 0; i < organs.size(); i++) {
            g_bestSessionVertices[i] = organs[i]->mVertices;
            g_bestSessionNormals[i]  = organs[i]->mNormals;
        }

        auto T = computeCurrentTransform();
        g_poseLibrary.saveCurrentToLibrary(
            method,
            g_sessionId,
            g_sessionBipopN,
            registrationHandle.refineCount,
            g_sessionSilhouetteN,
            elapsedSec,
            registrationHandle.fitness,
            registrationHandle.icpRmse,
            registrationHandle.averageError,
            registrationHandle.rmse,
            registrationHandle.maxError,
            registrationHandle.scaleFactor,
            registrationHandle.refineInitialRMSE,
            registrationHandle.refineBestRMSE,
            registrationHandle.refineBestIteration,
            registrationHandle.compRmse,
            registrationHandle.compAvgError,
            registrationHandle.compMaxError,
            registrationHandle.compIoU2D,
            registrationHandle.compCount,
            registrationHandle.compSource,
            registrationHandle.compTarget,
            T,
            g_currentOrientLabel,
            g_currentOrientRunCount);

    } else {
        std::cout << "[Session] Rejected by criterion=" << critName
                  << " : RMSE " << currentRmse << " (best " << g_bestSessionCompRmse << ")"
                  << ", IoU " << currentIoU << " (best " << g_bestSessionIoU2D << ")"
                  << " -> reverting" << std::endl;

        if (!g_bestSessionVertices.empty() &&
            g_bestSessionVertices.size() == organs.size()) {
            for (size_t i = 0; i < organs.size(); i++) {
                organs[i]->mVertices = g_bestSessionVertices[i];
                organs[i]->mNormals  = g_bestSessionNormals[i];
                setUp(*organs[i]);
            }
            registrationHandle.state = RegistrationData::REGISTERED;
            registrationHandle.useRegistration = true;
            computeUnifiedMetrics();
            std::cout << "[Session] Reverted. CompRMSE=" << registrationHandle.compRmse
                      << " IoU2D=" << registrationHandle.compIoU2D << std::endl;
        }
    }
}

static void poseApplyEntry(int entryId) {
    auto organs = getOrganList();
    if (g_poseLibrary.applyEntry(entryId, g_initOrganVertices, g_initOrganNormals, organs)) {
        registrationHandle.state = RegistrationData::REGISTERED;
        registrationHandle.useRegistration = true;
        float savedCompRmse = 0.0f;
        for (auto& e : g_poseLibrary.entries) {
            if (e.id == entryId) {
                savedCompRmse                        = e.compRmse;
                registrationHandle.fitness           = e.baseFitness;
                registrationHandle.icpRmse           = e.baseIcpRmse;
                registrationHandle.averageError      = e.baseAvgError;
                registrationHandle.rmse              = e.baseRmse;
                registrationHandle.maxError          = e.baseMaxError;
                registrationHandle.scaleFactor       = e.baseScale;
                registrationHandle.refineCount       = e.refineCount;
                registrationHandle.refineInitialRMSE   = e.refineInitialRMSE;
                registrationHandle.refineBestRMSE      = e.refineBestRMSE;
                registrationHandle.refineBestIteration = e.refineBestIteration;
                registrationHandle.compRmse          = e.compRmse;
                registrationHandle.compAvgError      = e.compAvgError;
                registrationHandle.compMaxError      = e.compMaxError;
                registrationHandle.compIoU2D         = e.compIoU2D;
                registrationHandle.compCount         = e.compCount;
                registrationHandle.compSource        = e.corrSource;
                registrationHandle.compTarget        = e.corrTarget;
                if (e.baseMethod == PoseEntry::FULL_AUTO)
                    gUIManager.state.regMethod = 0;
                else if (e.baseMethod == PoseEntry::HEMI_AUTO)
                    gUIManager.state.regMethod = 1;
                else if (e.baseMethod == PoseEntry::UMEYAMA)
                    gUIManager.state.regMethod = 2;
                else if (e.baseMethod == PoseEntry::BIPOP_CMAES)
                    gUIManager.state.regMethod = 3;
                else if (e.baseMethod == PoseEntry::SILHOUETTE_ALIGN)
                    gUIManager.state.regMethod = 5;
                else
                    gUIManager.state.regMethod = 1;
                break;
            }
        }
        computeUnifiedMetrics();
        float reproRmse = registrationHandle.compRmse;
        float diff = std::abs(reproRmse - savedCompRmse);
        std::cout << "[PoseLibrary] Reproduction check entry #" << entryId << std::endl;
        std::cout << "  Saved  CompRMSE: " << savedCompRmse << std::endl;
        std::cout << "  Repro  CompRMSE: " << reproRmse << std::endl;
        std::cout << "  Diff:            " << diff
                  << (diff < 1e-4f ? "  [OK]" : "  [WARN: drift detected]") << std::endl;

        /* Reset session-best to applied entry's values + snapshot current vertices.
           Without this, a future operation would falsely "degrade" against an old
           in-memory best from a different pose, causing unwanted reverts. */
        g_bestSessionCompRmse = registrationHandle.compRmse;
        g_bestSessionIoU2D    = registrationHandle.compIoU2D;
        g_bestSessionVertices.resize(organs.size());
        g_bestSessionNormals.resize(organs.size());
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) {
                g_bestSessionVertices[i] = organs[i]->mVertices;
                g_bestSessionNormals[i]  = organs[i]->mNormals;
            }
        }
        std::cout << "[Session] Reset session-best to applied entry: "
                  << "CompRMSE=" << g_bestSessionCompRmse
                  << " IoU2D=" << g_bestSessionIoU2D << std::endl;
    }
}

static void poseUndo() {
    auto organs = getOrganList();
    g_poseLibrary.undoToLast(g_initOrganVertices, g_initOrganNormals, organs);
    registrationHandle.state = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    computeUnifiedMetrics();
}

// -------------------------------------------------------
// Pose Library ImGui Window
// -------------------------------------------------------
static void drawPoseLibraryWindow() {
    if (!g_poseLibrary.showWindow) return;

    ImGui::SetNextWindowSize(ImVec2(640, 420), ImGuiCond_FirstUseEver);
    ImGui::PushStyleColor(ImGuiCol_WindowBg,      ImVec4(0.06f,0.06f,0.08f,0.95f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg,       ImVec4(0.12f,0.10f,0.18f,1.0f));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.20f,0.15f,0.30f,1.0f));

    if (ImGui::Begin("Pose Library", &g_poseLibrary.showWindow)) {
        ImGui::Text("Entries: %d / %d  |  Session #%d",
                    (int)g_poseLibrary.entries.size(), g_poseLibrary.maxEntries, g_sessionId);
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 248);
        {
            static bool s_importGuard = false;
            if (ImGui::Button("Import CSV", ImVec2(120,0)) && !s_importGuard) {
                s_importGuard = true;
#ifdef HAS_TINYFILEDIALOGS
                const char* filters[] = {"*.csv"};
                const char* sel = tinyfd_openFileDialog(
                    "Import Pose Library CSV","",1,filters,"CSV Files (*.csv)",0);
                if (sel) g_poseLibrary.importFromCsv(std::string(sel));
#else
                std::cerr << "[PoseLibrary] Build with -DHAS_TINYFILEDIALOGS for file picker." << std::endl;
#endif
            } else { s_importGuard = false; }
        }
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
        if (ImGui::Button("Export CSV", ImVec2(120,0))) {
            auto now = std::chrono::system_clock::now();
            auto tt  = std::chrono::system_clock::to_time_t(now);
            auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) % 1000;
            std::tm tm = *std::localtime(&tt);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "pose_library_%04d%02d%02d_%02d%02d%02d_%03d.csv",
                          tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
                          tm.tm_hour, tm.tm_min, tm.tm_sec, (int)ms.count());
            g_poseLibrary.exportToCsv(buf);
        }
        ImGui::Separator();

        // # | Session | Method | BIPOP | Refine | Silh | CompRMSE | IoU2D | N | Time | [Apply]
        ImGui::Columns(11, "pose_cols", true);
        ImGui::SetColumnWidth(0,  26);   // #
        ImGui::SetColumnWidth(1,  76);   // Session
        ImGui::SetColumnWidth(2,  60);   // Method (60 to fit "SilAln")
        ImGui::SetColumnWidth(3,  36);   // BIPOP
        ImGui::SetColumnWidth(4,  36);   // Refine
        ImGui::SetColumnWidth(5,  36);   // Silh
        ImGui::SetColumnWidth(6,  70);   // CompRMSE
        ImGui::SetColumnWidth(7,  60);   // IoU2D
        ImGui::SetColumnWidth(8,  40);   // N
        ImGui::SetColumnWidth(9,  44);   // Time
        ImGui::SetColumnWidth(10, 46);   // Apply

        auto hc = ImVec4(0.7f,0.7f,0.7f,1);
        ImGui::TextColored(hc, "#");        ImGui::NextColumn();
        ImGui::TextColored(hc, "Session");  ImGui::NextColumn();
        ImGui::TextColored(hc, "Method");   ImGui::NextColumn();
        ImGui::TextColored(hc, "BIPOP");    ImGui::NextColumn();
        ImGui::TextColored(hc, "Refine");   ImGui::NextColumn();
        ImGui::TextColored(hc, "Silh");     ImGui::NextColumn();
        ImGui::TextColored(hc, "CompRMSE"); ImGui::NextColumn();
        ImGui::TextColored(hc, "IoU2D");    ImGui::NextColumn();
        ImGui::TextColored(hc, "N");        ImGui::NextColumn();
        ImGui::TextColored(hc, "Time");     ImGui::NextColumn();
        ImGui::TextColored(hc, "");         ImGui::NextColumn();
        ImGui::Separator();

        int applyId = -1;
        int prevSessionId = -1;

        for (size_t i = 0; i < g_poseLibrary.entries.size(); i++) {
            auto& e = g_poseLibrary.entries[i];
            bool isActive = (e.id == g_poseLibrary.activeEntryId);

            if (prevSessionId >= 0 && e.sessionId != prevSessionId) {
                ImGui::Columns(1);
                ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.3f,0.3f,0.5f,0.8f));
                ImGui::Separator();
                ImGui::PopStyleColor();
                ImGui::Columns(11, "pose_cols", true);
                ImGui::SetColumnWidth(0,  26);
                ImGui::SetColumnWidth(1,  76);
                ImGui::SetColumnWidth(2,  60);
                ImGui::SetColumnWidth(3,  36);
                ImGui::SetColumnWidth(4,  36);
                ImGui::SetColumnWidth(5,  36);
                ImGui::SetColumnWidth(6,  70);
                ImGui::SetColumnWidth(7,  60);
                ImGui::SetColumnWidth(8,  40);
                ImGui::SetColumnWidth(9,  44);
                ImGui::SetColumnWidth(10, 46);
            }
            prevSessionId = e.sessionId;

            if (isActive) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f,1.0f,0.3f,1.0f));

            ImGui::Text("%d", (int)(i+1)); ImGui::NextColumn();

            ImGui::TextColored(ImVec4(0.55f,0.80f,1.0f,1.0f), "%s", e.sessionLabel().c_str());
            ImGui::NextColumn();

            {
                ImVec4 mc = ImVec4(0.85f,0.85f,0.85f,1);
                if (e.baseMethod == PoseEntry::HEMI_AUTO)        mc = ImVec4(0.94f,0.56f,0.19f,1);
                if (e.baseMethod == PoseEntry::BIPOP_CMAES)      mc = ImVec4(0.94f,0.56f,0.19f,1);
                if (e.baseMethod == PoseEntry::REFINE)           mc = ImVec4(0.13f,0.77f,0.37f,1);
                if (e.baseMethod == PoseEntry::UMEYAMA)          mc = ImVec4(0.55f,0.80f,1.0f,1);
                if (e.baseMethod == PoseEntry::SILHOUETTE_ALIGN) mc = ImVec4(0.85f,0.40f,0.95f,1);
                ImGui::TextColored(mc, "%s", e.methodStr());
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("ID: %d  Session: %s", e.id, e.sessionLabel().c_str());
                    ImGui::Text("Timestamp: %s", e.timestamp.c_str());
                    ImGui::Text("Elapsed: %.3f s", e.elapsedSec);
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(1,1,0.5f,1), "=== Unified Metrics ===");
                    ImGui::Text("Comp RMSE:   %.6f  (%d pairs)", e.compRmse, e.compCount);
                    ImGui::Text("Comp AvgErr: %.6f", e.compAvgError);
                    ImGui::Text("Comp MaxErr: %.6f", e.compMaxError);
                    if (e.compIoU2D > 0.0f)
                        ImGui::Text("Comp IoU2D:  %.4f", e.compIoU2D);
                    else
                        ImGui::Text("Comp IoU2D:  (not measured)");
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "--- Base Registration ---");
                    ImGui::Text("Fitness (ICP): %.6f", e.baseFitness);
                    ImGui::Text("ICP RMSE:      %.6f", e.baseIcpRmse);
                    ImGui::Text("Corr. RMSE:    %.6f", e.baseRmse);
                    ImGui::Text("Corr. AvgErr:  %.6f", e.baseAvgError);
                    ImGui::Text("Corr. MaxErr:  %.6f", e.baseMaxError);
                    ImGui::Text("Scale:         %.4f", e.baseScale);
                    if (e.refineCount > 0) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(0.13f,0.77f,0.37f,1), "--- Refine ---");
                        ImGui::Text("Count:     %d",   e.refineCount);
                        ImGui::Text("Init RMSE: %.6f", e.refineInitialRMSE);
                        ImGui::Text("Best RMSE: %.6f", e.refineBestRMSE);
                        ImGui::Text("Best Iter: %d",   e.refineBestIteration);
                    }
                    if (e.silhouetteCount > 0) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(0.85f,0.40f,0.95f,1), "--- Silhouette ---");
                        ImGui::Text("Count: %d",   e.silhouetteCount);
                        ImGui::Text("IoU:   %.4f", e.compIoU2D);
                    }
                    ImGui::EndTooltip();
                }
            }
            ImGui::NextColumn();

            if (e.bipopCount > 0)
                ImGui::TextColored(ImVec4(0.94f,0.56f,0.19f,0.9f), "x%d", e.bipopCount);
            else
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "---");
            ImGui::NextColumn();

            if (e.refineCount > 0)
                ImGui::TextColored(ImVec4(0.13f,0.77f,0.37f,0.9f), "x%d", e.refineCount);
            else
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "---");
            ImGui::NextColumn();

            if (e.silhouetteCount > 0)
                ImGui::TextColored(ImVec4(0.85f,0.40f,0.95f,0.9f), "x%d", e.silhouetteCount);
            else
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "---");
            ImGui::NextColumn();

            ImGui::Text("%.4f", e.compRmse); ImGui::NextColumn();

            if (e.compIoU2D > 0.0f) {
                /* IoU 高いほど緑寄り、低いほどグレー寄り */
                float t = std::min(1.0f, std::max(0.0f, (e.compIoU2D - 0.7f) / 0.3f));
                ImVec4 ic = ImVec4(0.55f + (0.0f - 0.55f) * t,
                                   0.55f + (0.85f - 0.55f) * t,
                                   0.55f + (0.30f - 0.55f) * t, 1.0f);
                ImGui::TextColored(ic, "%.3f", e.compIoU2D);
            } else {
                ImGui::TextColored(ImVec4(0.3f,0.3f,0.3f,1), "—");
            }
            ImGui::NextColumn();

            ImGui::Text("%d",   e.compCount); ImGui::NextColumn();

            {
                char tbuf[16];
                if (e.elapsedSec < 100.0f)
                    std::snprintf(tbuf, sizeof(tbuf), "%.1fs", e.elapsedSec);
                else
                    std::snprintf(tbuf, sizeof(tbuf), "%ds", (int)e.elapsedSec);
                ImGui::TextColored(ImVec4(0.6f,0.6f,0.6f,1), "%s", tbuf);
            }
            ImGui::NextColumn();

            ImGui::PushID(e.id);
            if (ImGui::SmallButton("Apply")) applyId = e.id;
            ImGui::PopID();
            ImGui::NextColumn();

            if (isActive) ImGui::PopStyleColor();
        }

        ImGui::Columns(1);

        if (applyId >= 0) poseApplyEntry(applyId);

        ImGui::Separator();
        ImGui::Spacing();

        float bw = (ImGui::GetContentRegionAvail().x - 8) / 2.0f;
        bool canUndo = g_poseLibrary.hasLastRegistration;
        if (!canUndo) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.4f);
        if (ImGui::Button("Undo", ImVec2(bw, 28))) { if (canUndo) poseUndo(); }
        if (!canUndo) ImGui::PopStyleVar();
        ImGui::SameLine();
        if (ImGui::Button("Clear All", ImVec2(bw, 28))) {
            g_poseLibrary.entries.clear();
            g_poseLibrary.activeEntryId = -1;
        }
    }
    ImGui::End();
    ImGui::PopStyleColor(3);
}

/* Shift+E = 2D IoU BIPOP-CMA-ES (高速ラスタライズ版)
   Shift+V (3D compRMSE BIPOP) の 2D 版。異なる sigma0 と初期摂動で
   N 回 CMA-ES を回し、最良 IoU の結果を採択する。
   単発だと scale ≒ 1.0 付近のローカル最適 (例: IoU 0.89)
   から抜け出せないケースに有効。
   ワークフロー: HemiAuto ボタン → Shift+V → Shift+E で仕上げ。

   実装履歴:
   - 旧 Shift+E (Raycast 版) と旧 Ctrl+Shift+E (Fast 版) を比較検証し、
     両者は |IoU_diff|=0.000000 で完全一致することを確認済み。
   - Fast 版が 2-3 倍高速かつ数値的に同等のため、Fast 版を本命採用。
   - 同じシード + 同じ初期姿勢なら結果は再現可能 (決定論的)。
   - Raycast 版の関数は CmaesUtils.h に残置 (将来の検証用)。

   呼び出し元: キーボード Shift+E / UI ボタン "Silhouette Alignment" */
static void runShiftE() {
    if (currentMainMode != REGISTRATION_MODE) return;
    const std::string keyLabel = "Shift+E";

    std::cout << "\n=== 2D Silhouette BIPOP-CMA-ES (" << keyLabel
              << ", Fast Rasterize) ===" << std::endl;
    if (registrationHandle.compRmse == 0.0f) {
        std::cerr << "[" << keyLabel << "] No registration yet."
                  << " Run HemiAuto (button) first." << std::endl;
        return;
    }
    poseAutoSaveBeforeRegistration();
    auto organs = getOrganList();
    computeUnifiedMetrics();
    float compRmse_before = registrationHandle.compRmse;

    /* 現在のカメラに対応する target を用意する。
       ユーザーが初期カメラのままなら legacy shortcut と等価、
       カメラをスクロールしている場合は screenMesh 上の SAM2 mask
       を現カメラから再 render して target にする。これにより
       どのカメラからでも「画面上で mask に合わせる」意味の
       最適化が走る。CMA-ES ループ中は camera/screenMesh 不変
       なので 1 回 build すれば 300 × 5 eval 全て再利用される。
       Shift+E 終了時に invalidate するので、カメラを動かした
       あとの別 metric 計算では自動的に legacy shortcut に戻る。

       重要: HemiAuto ボタンは同期実行で main loop を止めるため、
       その直後グローバル view/projection が stale な可能性。
       ここで OrbitCam.UpdateCamera() を呼んで強制最新化し、
       OrbitCam.view / OrbitCam.projection を直接渡す。
       model は organ の世界座標系がそのまま使われるので
       identity でよい (グローバル model も objPos translate のみ)。 */
    std::cout << "[Shift+E] preparing buildProjectedLiverMask"
              << std::endl;
    std::cout << "[Shift+E] global view==identity? "
              << (view == glm::mat4(1.0f) ? "YES (stale!)" : "no")
              << "  global proj==identity? "
              << (projection == glm::mat4(1.0f) ? "YES (stale!)" : "no")
              << std::endl;
    OrbitCam.UpdateCamera();   /* refresh OrbitCam.view/projection
                                  + push to global view/projection
                                  via setGlobalMatrixPointers */
    std::cout << "[Shift+E] post-UpdateCamera: view==identity? "
              << (OrbitCam.view == glm::mat4(1.0f) ? "YES" : "no")
              << "  proj==identity? "
              << (OrbitCam.projection == glm::mat4(1.0f) ? "YES" : "no")
              << std::endl;
    buildProjectedLiverMask(OrbitCam.view, OrbitCam.projection,
                            model, OrbitCam.cameraPos);

    /* 現在の 2D IoU を測定 (Fast ラスタライズ版)。
       BIPOP ループ内で最良姿勢を判定する指標に使うため、
       CMA-ES 内部目的関数と同じ実装を使う。 */
    auto measureIoU = [&]() -> float {
        float fval = CmaesRefine::computeSilhouette2DObjectiveFast(
            liverMesh3D, view, projection, 8);
        return 1.0f - fval;
    };
    float iou_before = measureIoU();
    std::cout << "[" << keyLabel << "] Initial IoU=" << iou_before
              << "  compRMSE=" << compRmse_before
              << " (compRMSE is informational only)" << std::endl;

    /* 現在の頂点をスナップショット (run 間で戻す起点) */
    std::vector<std::vector<GLfloat>> start_v(organs.size());
    std::vector<std::vector<GLfloat>> start_n(organs.size());
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            start_v[i] = organs[i]->mVertices;
            start_n[i] = organs[i]->mNormals;
        }
    }

    /* ベスト姿勢 (IoU 最大) を保持 */
    float best_iou = iou_before;
    std::vector<std::vector<GLfloat>> best_v = start_v;
    std::vector<std::vector<GLfloat>> best_n = start_n;

    const int N_STARTS = 5;
    /* Shift+E と Ctrl+Shift+E を完全再現可能にするため、外側 BIPOP の
       rng を固定シードで初期化する。
       これにより、同じ初期姿勢から始めれば Raycast 版と Fast 版は
       各 run で同じ摂動・同じ sigma0・同じ CMA-ES 軌道を辿り、
       結果 IoU も一致するはず (Fast 版と Raycast 版は既に同一出力
       であることが比較デバッグで確認済み)。
       実験的に異なるシード試したいときはここを変更。 */
    const uint32_t SHIFT_E_BIPOP_SEED = 20260420u;
    std::mt19937 rng(SHIFT_E_BIPOP_SEED);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int run = 0; run < N_STARTS; run++) {
        /* 毎回 start_v に戻す */
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) {
                organs[i]->mVertices = start_v[i];
                organs[i]->mNormals  = start_n[i];
                setUp(*organs[i]);
            }
        }

        CmaesRefine::Params p;
        p.verbose        = true;
        p.log_every      = 100;
        p.save_debug_jpg = false;
        p.maxgen         = 300;
        p.tx_range = 1.0f; p.ty_range = 1.0f; p.tz_range = 1.0f;
        p.rx_range = 20.0f; p.ry_range = 20.0f; p.rz_range = 20.0f;
        p.scale_lo = 0.85f; p.scale_hi = 1.15f;
        /* 2D IoU を目的関数として使う */
        p.use_silhouette_2d = true;
        p.use_silhouette_2d_fast = true; /* Fast 版を本命採用 */
        p.alpha_silhouette  = 1.0f;
        p.alpha_3d          = 0.3f;
        /* 高速化: ラスタライズ解像度 step=4→8 (ピクセル数 1/4) */
        p.silhouette_step   = 8;
        /* 高速化: tolfun 1e-6→1e-4 (CMA-ES 早期終了) */
        p.tolfun            = 1e-4;
        /* CMA-ES 内部のシード固定 (run ごとに異なる値、ただし決定論的) */
        p.rng_seed = SHIFT_E_BIPOP_SEED + 1000u * (uint32_t)(run + 1);

        float tx_perturb = 0.0f, ty_perturb = 0.0f, tz_perturb = 0.0f;
        float rx_perturb = 0.0f, ry_perturb = 0.0f, rz_perturb = 0.0f;
        float sc_perturb = 1.0f;
        std::string regime;

        if (run == 0) {
            /* Run 0: 摂動なし・sigma0 小 (精密ベースライン) */
            p.sigma0 = 0.2;
            regime = "Baseline";
        } else if (run <= 2) {
            /* Run 1-2: 中 sigma0 + 中摂動 (局所探索) */
            p.sigma0 = 0.05f + dist01(rng) * 0.25f;
            tx_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
            ty_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
            tz_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
            rx_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
            ry_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
            rz_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
            sc_perturb = 0.95f + dist01(rng) * 0.10f;
            regime = "Local";
        } else {
            /* Run 3-4: 大 sigma0 + 大摂動 (scale の盆地を越えるため
               sc_perturb の範囲を 0.85〜1.15 まで広げる) */
            p.sigma0 = 0.30f + dist01(rng) * 0.50f;
            tx_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
            ty_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
            tz_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
            rx_perturb = (dist01(rng)*2.0f-1.0f) * 20.0f;
            ry_perturb = (dist01(rng)*2.0f-1.0f) * 20.0f;
            rz_perturb = (dist01(rng)*2.0f-1.0f) * 20.0f;
            sc_perturb = 0.85f + dist01(rng) * 0.30f; /* 0.85〜1.15 */
            regime = "Global";
        }

        /* 初期摂動を適用 */
        if (run > 0) {
            CmaesRefine::applyIncrementalSRT(organs,
                                             tx_perturb, ty_perturb, tz_perturb,
                                             rx_perturb, ry_perturb, rz_perturb,
                                             sc_perturb);
            for (size_t i = 0; i < organs.size(); i++)
                if (organs[i]) setUp(*organs[i]);
        }

        std::cout << "[" << keyLabel << "] Run " << (run+1) << "/" << N_STARTS
                  << "  " << regime
                  << "  sigma0=" << std::fixed << std::setprecision(2) << p.sigma0
                  << "  sc_perturb=" << sc_perturb
                  << std::endl;

        CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                                 gGridWidth, gGridHeight(),
                                                 gDepthScale, p);

        /* この run 終了時点の IoU を測定 (CMA-ES 内で accept 済み or revert 済み) */
        float iou_run = measureIoU();
        std::cout << "[" << keyLabel << "] Run " << (run+1)
                  << "  final IoU=" << std::setprecision(6) << iou_run
                  << (iou_run > best_iou + 1e-4f ? " [NEW BEST]" : "")
                  << std::endl;

        if (iou_run > best_iou + 1e-4f) {
            best_iou = iou_run;
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i]) {
                    best_v[i] = organs[i]->mVertices;
                    best_n[i] = organs[i]->mNormals;
                }
            }
        }
    }

    /* ベスト姿勢を適用 */
    for (size_t i = 0; i < organs.size(); i++) {
        if (organs[i]) {
            organs[i]->mVertices = best_v[i];
            organs[i]->mNormals  = best_n[i];
            setUp(*organs[i]);
        }
    }
    computeUnifiedMetrics();
    float iou_after = measureIoU();
    std::cout << "[" << keyLabel << "] === BIPOP Summary ===" << std::endl;
    std::cout << "[" << keyLabel << "] IoU: " << iou_before << " -> " << iou_after
              << (iou_after > iou_before + 1e-4f ? " [IMPROVED]" : " [NO CHANGE]")
              << std::endl;
    std::cout << "[" << keyLabel << "] compRMSE: " << compRmse_before
              << " -> " << registrationHandle.compRmse
              << " (informational)" << std::endl;

    /* デバッグ画像を最終姿勢で保存。
       silhouette_final_shiftE_{target,source,composite}.jpg
         赤=target マスク輪郭, 青=source (レイキャスト抽出 輪郭)
       silhouette_final_shiftE_rast_{target,source,composite}.jpg
         赤=target, 緑=source (ラスタライズ結果からの輪郭)
       両者ほぼ同一 (既に互換性証明済み) */
    {
        const std::string prefix = std::string(DEPTH_OUTPUT_PATH)
                                   + "silhouette_final_shiftE";
        CmaesRefine::saveSilhouetteDebugJPG(liverMesh3D, view, projection,
                                            prefix, 0.3f);
        CmaesRefine::saveSilhouetteFastDebugJPG(liverMesh3D, view, projection,
                                                 prefix + "_rast", 8);
        std::cout << "[" << keyLabel
                  << "] Debug JPGs saved with prefix: " << prefix << std::endl;
    }

    /* Pose Library 保存 (SaveCriterion::IOU で IoU 改善時のみ accept)。
       compRMSE が劣化していても IoU が改善していれば accept される。
       g_bestSessionVertices は呼び出し先で更新されるので revert anchor も移動する。 */
    g_sessionSilhouetteN++;
    gUIManager.state.regMethod = 5;
    poseSaveToLibrary(SaveCriterion::IOU);

    /* Shift+T (Ctrl+Shift+E) 軌道検索の端点として最終姿勢を保存 */
    if (!g_initOrganVertices.empty() && liverMesh3D &&
        !g_initOrganVertices[0].empty() &&
        g_initOrganVertices[0].size() == liverMesh3D->mVertices.size()) {
        gShiftE_lastTransform = PoseLibrary::computeTransformFromLiver(
            g_initOrganVertices[0], liverMesh3D->mVertices);
        gShiftE_lastValid = true;
        std::cout << "[Shift+T] Shift+E endpoint saved (gShiftE_lastTransform)" << std::endl;
    }

    /* 最適化完了。ユーザーがカメラを動かしても旧 target が
       使い回されないよう invalidate。次回 Shift+E 時にその時の
       カメラで改めて build される。 */
    g_projectedLiverMask.invalidate();
}

void setupUICallbacks() {
    auto& a = gUIManager.actions;

    a.onToggleCamera = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        if (gCameraPreview.active) {
            clearSegPoints();
            gCameraPreview.captureAndFreeze(screenMesh);
            depthSplitScreenMode = true;
            splitScreenMode = true;
            OrbitCamLeft_Target = OrbitCam;
            OrbitCamRight_Screen = OrbitCam;
            OrbitCamLeft_Target.currentTarget = TARGET_LIVER;
            OrbitCamLeft_Target.cx = (gWindowWidth / 2) / 2.0f;
            OrbitCamLeft_Target.cy = gWindowHeight / 2.0f;
            OrbitCamRight_Screen.currentTarget = TARGET_TEXTURE;
            OrbitCamRight_Screen.gRadius = OrbitCam.InitialRadius * 2.0f;
            OrbitCamRight_Screen.cx = (gWindowWidth / 2) / 2.0f;
            OrbitCamRight_Screen.cy = gWindowHeight / 2.0f;
        } else if (gCameraPreview.frozen) {
            depthSplitScreenMode = false;
            splitScreenMode = false;
            gCameraPreview.clearFrozen();
            clearSegPoints();
            gCameraPreview.start(screenMesh, 0, 1280, 720);
        } else {
            clearSegPoints();
            gCameraPreview.start(screenMesh, 0, 1280, 720);
        }
    };

    a.onRunDepth = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        showProgressOverlay(0.05f, "Preparing depth...");
        std::vector<DepthRunnerPoint> segPoints;
        if (gUserSegPoints.empty()) {
            segPoints = createDefaultSegPoints(
                screenMesh->loadedImageWidth, screenMesh->loadedImageHeight);
        } else {
            segPoints = gUserSegPoints;
        }
        if (gCameraPreview.frozen) {
            gCameraPreview.runDepthFromFrozen(gDepthRunner, screenMesh, segPoints, showProgressOverlay);
        } else if (gCameraPreview.active) {
            gCameraPreview.captureAndRunDepthWithPoints(gDepthRunner, screenMesh, segPoints, showProgressOverlay);
        } else if (gDepthRunner.isAvailable()) {
            DepthRunnerIntegration::updateScreenMeshDepth(
                gDepthRunner, gDepthInputImage, screenMesh,
                128, 10.0f, 0.3f, segPoints,
                [](mCutMesh& mesh) { setUp(mesh); },
                showProgressOverlay);
        }
        showProgressOverlay(1.0f, "Depth complete!");
        resetBoundaryMap();
        gDepthScale = 0.3f;
        clearSegPoints();
        depthSplitScreenMode = false;
        splitScreenMode = false;
        registrationHandle.reset();
        registrationHandle.state = RegistrationData::IDLE;
        g_refineVertexIndices.clear();
        g_cluster1Points.clear();
        g_cluster2Points.clear();
        g_targetPoints.clear();
        g_showClusterVisualization = false;
        g_showCorrespondencePoints = false;
    };

    a.onResetDefaultImage = []() {
        gCameraPreview.stop();
        gCameraPreview.clearFrozen();
        clearSegPoints();
        gDepthScale = 0.3f;
        gDroppedFilePath = "";
        depthSplitScreenMode = false;
        splitScreenMode = false;
        showProgressOverlay(0.05f, "Resetting to default...");
        initScreenMeshWithDepthRunner(gDepthRunner, screenMesh, false, showProgressOverlay);
        showProgressOverlay(1.0f, "Reset complete!");
    };

    a.onLoadLocalImage = []() {
        if (currentMainMode == REGISTRATION_MODE) openImageFilePicker();
    };

    a.onUndoSegPoint = []() {
        if (!gUserSegPoints.empty()) {
            gUserSegPoints.pop_back();
            gUserSegPoints3D.pop_back();
            gUserSegPointsFG.pop_back();
        }
    };

    a.onDepthScaleChanged = [](float v) {
        gDepthScale = v;
        regenerateDepthMeshAuto(screenMesh, gDepthScale, gMeshScale);
        if (g_showClusterVisualization && registrationHandle.state == RegistrationData::REGISTERED) {
            computeUnifiedMetrics();
        }
    };

    a.onFullAuto = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        gUIManager.state.regMethod = 0;
        splitScreenMode = false;
        poseAutoSaveBeforeRegistration();
        registrationHandle.reset();
        registrationHandle.state = RegistrationData::IDLE;
        OrbitCam.cx = gWindowWidth / 2.0f;
        OrbitCam.cy = gWindowHeight / 2.0f;

        auto organs = getOrganList();
        std::vector<std::string> names = {"Liver","Portal","Vein","Tumor","Segment","Gallbladder"};
        Reg3DCustom::performRegistrationMultiMeshWithScale(
            organs, names, screenMesh, OrbitCam.cameraPos,
            gGridWidth, gGridHeight(), 15, 0.005f, 0.35f, true, 0.03f, gDepthScale);
        computeUnifiedMetrics();
        poseSaveToLibrary();
    };

    a.onHemiAuto = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        gUIManager.state.regMethod = 1;
        g_stepStartTime  = std::chrono::steady_clock::now();
        g_sessionBipopN  = 0;
        g_sessionRefineN = 0;
        poseAutoSaveBeforeRegistration();
        resetRegistrationState();

        Reg3D::BVHTree bvh;
        bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
        auto vis = Reg3DCustom::extractVisibleVerticesCustom(
            *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);
        if (vis.cloud->size() < 50) return;
        g_cluster1Points = vis.points;
        g_cluster2Points.clear();
        g_refineVertexIndices = vis.vertexIndices;
        computeIdealVoxelSizes();
        auto organs = getOrganList();
        Reg3DCustom::performRegistrationSingleMesh(
            organs, liverMesh3D, vis.vertexIndices,
            screenMesh, OrbitCam.cameraPos,
            gGridWidth, gGridHeight(), 15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);
        computeUnifiedMetrics();
        poseSaveToLibrary();
    };

    a.onHemiVoxelChanged = [](float v) {
        g_voxelSize = v;
    };

    a.onInitRotPresetChanged = [](int preset) {
        startNewSession();
        registrationHandle.initRotPreset =
            (RegistrationData::InitRotPreset)preset;
        std::cout << "[InitRot] Preset selected: "
                  << RegistrationData::presetName(registrationHandle.initRotPreset)
                  << std::endl;

        g_currentOrientLabel = RegistrationData::presetName(registrationHandle.initRotPreset);
        std::cout << "[Session] New session: " << g_currentOrientLabel << std::endl;

        auto organs = getOrganList();
        if (g_initOrganVertices.empty() ||
            g_initOrganVertices.size() != organs.size()) return;

        for (size_t i = 0; i < organs.size(); i++) {
            organs[i]->mVertices = g_initOrganVertices[i];
            organs[i]->mNormals  = g_initOrganNormals[i];
            setUp(*organs[i]);
        }

        RegistrationData::InitRotPreset p =
            (RegistrationData::InitRotPreset)preset;
        if (p != RegistrationData::PRESET_FRONT) {
            glm::vec3 centroid =
                computeMeshCentroidFromVertices(liverMesh3D->mVertices);
            glm::mat4 R = getPresetRotation(p, centroid);
            for (auto* m : organs) {
                applyMatrixToMeshVerticesAndNormals(m, R);
                setUp(*m);
            }
        }
    };

    a.onRefine = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        if (registrationHandle.state == RegistrationData::REGISTERED) {
            if (g_refineVertexIndices.empty()) {
                std::cerr << "[Refine] No visible vertex indices." << std::endl;
                return;
            }
            g_stepStartTime = std::chrono::steady_clock::now();
            poseAutoSaveBeforeRegistration();
            std::cout << "\n=== Normal-Compatible Refinement START ===" << std::endl;
            std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                              tumorMesh3D, segmentMesh3D, gbMesh3D};
            NormalRefine::RefineParams params;
            params.useZWeight      = true;
            params.zWeightBoundary = 0.05f;
            params.zWeightInterior = 0.30f;
            params.boundaryWidth   = 8.0f;
            params.boundaryBoost   = 3.0f;
            if (NormalRefine::initRefine(g_refineState, liverMesh3D,
                                         g_refineVertexIndices,
                                         screenMesh, organs,
                                         gGridWidth, gGridHeight(), gDepthScale, params,
                                         NormalRefine::NORMAL_COMPAT)) {
                // Override initial/best RMSE with unified Target→Source metric
                computeUnifiedMetrics();
                g_refineState.initialRMSE = registrationHandle.compRmse;
                g_refineState.bestRMSE    = registrationHandle.compRmse;
                std::cout << "[Refine] Unified initial RMSE: " << registrationHandle.compRmse << std::endl;
                registrationHandle.state = RegistrationData::REFINING;
            } else {
                std::cerr << "[Refine] Initialization failed" << std::endl;
            }
        } else if (registrationHandle.state == RegistrationData::REFINING) {
            g_refineState.active = false;
            registrationHandle.state = RegistrationData::REGISTERED;
            bool improved = g_refineState.bestRMSE < g_refineState.initialRMSE;
            g_refineState.restoreMeshes();
            if (improved) {
                NormalRefine::applyIncrementalTransform(
                    g_refineState.bestCumulativeTransform,
                    g_refineState.organMeshes);
            }
            // Update refine metrics in registrationHandle
            registrationHandle.refineCount++;
            registrationHandle.refineInitialRMSE   = g_refineState.initialRMSE;
            registrationHandle.refineBestRMSE      = g_refineState.bestRMSE;
            registrationHandle.refineBestIteration = g_refineState.bestIteration;
            computeUnifiedMetrics();
            poseSaveToLibrary();
            std::cout << "\n=== Refinement STOPPED ===" << std::endl;
            std::cout << "  Initial RMSE: " << g_refineState.initialRMSE << std::endl;
            std::cout << "  Best RMSE:    " << g_refineState.bestRMSE
                      << " (iter " << g_refineState.bestIteration << ")" << std::endl;
            if (improved)
                std::cout << "  >> Reverted to best state" << std::endl;
            else
                std::cout << "  >> No improvement — reverted to initial" << std::endl;
        }
    };

    a.onBipopCmaes = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        if (registrationHandle.compRmse == 0.0f) {
            std::cerr << "[UI] No registration yet. Run HemiAuto first." << std::endl;
            return;
        }
        g_stepStartTime = std::chrono::steady_clock::now();
        g_sessionBipopN++;
        gUIManager.state.regMethod = 3;
        std::cout << "\n=== BIPOP-CMA-ES Multi-Start (UI Button) ===" << std::endl;
        poseAutoSaveBeforeRegistration();
        auto organs = getOrganList();
        computeUnifiedMetrics();
        float rmse_before = registrationHandle.compRmse;
        std::cout << "[Shift+V] Current compRMSE: " << rmse_before << std::endl;

        std::vector<std::vector<GLfloat>> start_v(organs.size());
        std::vector<std::vector<GLfloat>> start_n(organs.size());
        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) { start_v[i] = organs[i]->mVertices; start_n[i] = organs[i]->mNormals; }
        }

        float best_rmse = rmse_before;
        std::vector<std::vector<GLfloat>> best_v = start_v;
        std::vector<std::vector<GLfloat>> best_n = start_n;

        const int N_STARTS = 10;
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        for (int run = 0; run < N_STARTS; run++) {
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i]) { organs[i]->mVertices = start_v[i]; organs[i]->mNormals = start_n[i]; setUp(*organs[i]); }
            }

            CmaesRefine::Params p;
            p.verbose        = true;
            p.log_every      = 100;
            p.save_debug_jpg = false;

            float tx_perturb = 0, ty_perturb = 0, tz_perturb = 0;
            float rx_perturb = 0, ry_perturb = 0, rz_perturb = 0;
            float sc_perturb = 1.0f;
            std::string regime;

            if (run % 2 == 0) {
                p.sigma0 = 0.3 + dist01(rng) * 0.4;
                tx_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
                ty_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
                tz_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
                rx_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
                ry_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
                rz_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
                sc_perturb = 0.95f + dist01(rng) * 0.10f;
                regime = "Regime2(local)";
            } else {
                p.sigma0 = 0.5 + dist01(rng) * 0.5;
                tx_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
                ty_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
                tz_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
                rx_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
                ry_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
                rz_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
                sc_perturb = 0.90f + dist01(rng) * 0.20f;
                regime = "Regime1(global)";
            }

            if (run > 0) {
                CmaesRefine::applyIncrementalSRT(organs, tx_perturb, ty_perturb, tz_perturb,
                                                 rx_perturb, ry_perturb, rz_perturb, sc_perturb);
                for (size_t i = 0; i < organs.size(); i++) if (organs[i]) setUp(*organs[i]);
            }

            std::cout << "[BIPOP] Run " << (run+1) << "/" << N_STARTS << "  " << regime
                      << "  sigma0=" << std::fixed << std::setprecision(2) << p.sigma0 << std::endl;

            CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                                     gGridWidth, gGridHeight(), gDepthScale, p);
            computeUnifiedMetrics();
            float rmse_run = registrationHandle.compRmse;
            std::cout << "[BIPOP] Run " << (run+1) << " compRMSE=" << std::setprecision(6) << rmse_run
                      << (r.improved ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;

            if (rmse_run < best_rmse) {
                best_rmse = rmse_run;
                for (size_t i = 0; i < organs.size(); i++) {
                    if (organs[i]) { best_v[i] = organs[i]->mVertices; best_n[i] = organs[i]->mNormals; }
                }
            }
        }

        for (size_t i = 0; i < organs.size(); i++) {
            if (organs[i]) { organs[i]->mVertices = best_v[i]; organs[i]->mNormals = best_n[i]; setUp(*organs[i]); }
        }
        computeUnifiedMetrics();
        std::cout << "[BIPOP] Best: " << rmse_before << " -> " << best_rmse
                  << (best_rmse < rmse_before ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;
        poseSaveToLibrary();

        /* Shift+T (Ctrl+Shift+E) 軌道検索の端点として最終姿勢を保存 */
        if (!g_initOrganVertices.empty() && liverMesh3D &&
            !g_initOrganVertices[0].empty() &&
            g_initOrganVertices[0].size() == liverMesh3D->mVertices.size()) {
            gShiftV_lastTransform = PoseLibrary::computeTransformFromLiver(
                g_initOrganVertices[0], liverMesh3D->mVertices);
            gShiftV_lastValid = true;
            std::cout << "[Shift+T] BIPOP endpoint saved (gShiftV_lastTransform)" << std::endl;
        }
    };

    a.onSilhouetteAlign = []() { runShiftE(); };

    a.onStartUmeyama = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        gUIManager.state.regMethod = 2;
        isDragging = false; hit_index = -1;
        if (splitScreenMode) splitScreenMode = false;
        registrationHandle.reset();
        registrationHandle.targetPointCount = 5;
        registrationHandle.state = RegistrationData::SELECTING_BOARD_POINTS;
        registrationHandle.useRegistration = false;
        OrbitCam.resetToInitialState();
        OrbitCam.cx = gWindowWidth / 2.0f;
        OrbitCam.cy = gWindowHeight / 2.0f;
        OrbitCamLeft_Target = OrbitCam;
        OrbitCamRight_Screen = OrbitCam;
        OrbitCamLeft_Target.gRadius = OrbitCam.InitialRadius * 1.0f;
        OrbitCamLeft_Target.currentTarget = TARGET_LIVER;
        OrbitCamLeft_Target.cx = (gWindowWidth / 2) / 2.0f;
        OrbitCamLeft_Target.cy = gWindowHeight / 2.0f;
        OrbitCamRight_Screen.gRadius = OrbitCam.InitialRadius * 2.0f;
        OrbitCamRight_Screen.currentTarget = TARGET_TEXTURE;
        OrbitCamRight_Screen.cx = (gWindowWidth / 2) / 2.0f;
        OrbitCamRight_Screen.cy = gWindowHeight / 2.0f;
        splitScreenMode = true;
    };

    a.onExecuteUmeyama = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        if (!registrationHandle.canRegister()) return;
        poseAutoSaveBeforeRegistration();
        std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                          tumorMesh3D, segmentMesh3D, gbMesh3D};
        performRegistrationUmeyama(registrationHandle, organs);
        computeUnifiedMetrics();
        poseSaveToLibrary();
        splitScreenMode = false;
        OrbitCam = OrbitCamRight_Screen;
        OrbitCam.gRadius = OrbitCam.InitialRadius;
        OrbitCam.cx = gWindowWidth / 2.0f;
        OrbitCam.cy = gWindowHeight / 2.0f;
    };

    a.onResetRegistration = []() {
        startNewSession();
        registrationHandle.resetTransformOnly();
        splitScreenMode = false;
        gUIManager.state.regMethod = -1;
        g_refineVertexIndices.clear();
        g_refineState.reset();

        auto organs = getOrganList();
        if (!g_initOrganVertices.empty() &&
            g_initOrganVertices.size() == organs.size()) {
            for (size_t i = 0; i < organs.size(); i++) {
                organs[i]->mVertices = g_initOrganVertices[i];
                organs[i]->mNormals  = g_initOrganNormals[i];
                setUp(*organs[i]);
            }
        }

        registrationHandle.initRotPreset = RegistrationData::PRESET_FRONT;
        gUIManager.state.initRotPreset   = 0;
        g_currentOrientLabel             = "Front";
        std::cout << "[InitRot] Reset to Front" << std::endl;
    };

    a.onPoseUndo = []() {
        poseUndo();
    };

    a.onSwitchDepthModel = [](int idx) {
        switchDepthModel(gDepthRunner, idx);
        gUIManager.state.depthModelIdx = idx;
    };

    a.onPoseLibraryToggle = []() {
        g_poseLibrary.showWindow = !g_poseLibrary.showWindow;
    };

    a.onClearPoints = []() {
        registrationHandle.clearPoints();
        registrationHandle.state = RegistrationData::IDLE;
        g_showCorrespondencePoints = false;
        std::cout << "Correspondence points cleared" << std::endl;
    };

    a.onToggleCorrespondenceVis = []() {
        g_showCorrespondencePoints = !g_showCorrespondencePoints;
        std::cout << "Correspondence points: "
                  << (g_showCorrespondencePoints ? "ON" : "OFF") << std::endl;
    };

    a.onUndoUmeyamaPoint = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        if (registrationHandle.state == RegistrationData::READY_TO_REGISTER ||
            registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS) {
            if (!registrationHandle.objectPoints.empty()) {
                registrationHandle.objectPoints.pop_back();
                std::cout << "[Umeyama] Undo object point. Remaining: "
                          << registrationHandle.objectPoints.size() << std::endl;
                if (registrationHandle.state == RegistrationData::READY_TO_REGISTER)
                    registrationHandle.state = RegistrationData::SELECTING_OBJECT_POINTS;
                return;
            }
        }
        if (registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS &&
            registrationHandle.objectPoints.empty()) {
            if (!registrationHandle.boardPoints.empty()) {
                registrationHandle.boardPoints.pop_back();
                registrationHandle.state = RegistrationData::SELECTING_BOARD_POINTS;
                std::cout << "[Umeyama] Undo board point (back to board phase). Remaining: "
                          << registrationHandle.boardPoints.size() << std::endl;
            }
            return;
        }
        if (registrationHandle.state == RegistrationData::SELECTING_BOARD_POINTS) {
            if (!registrationHandle.boardPoints.empty()) {
                registrationHandle.boardPoints.pop_back();
                std::cout << "[Umeyama] Undo board point. Remaining: "
                          << registrationHandle.boardPoints.size() << std::endl;
            }
        }
    };

    a.onRigidMode = []() {
        if (currentMainMode != DEFORM_MODE) return;
        deformHandlPlace.state = DeformHandlPlaceData::RIGID_MODE;
        if (multiBody) multiBody->setRigidMode(true);
    };

    a.onHandlePlaceMode = []() {
        if (currentMainMode != DEFORM_MODE) return;
        if (multiBody) {
            multiBody->setRigidMode(true);
            multiBody->initPhysics();
            multiBody->reapplyHandleConstraints();
        }
        deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
    };

    a.onDeformMode = []() {
        if (currentMainMode != DEFORM_MODE) return;
        if (multiBody) multiBody->setRigidMode(false);
        deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
    };

    a.onFullReset = []() {
        if (currentMainMode != DEFORM_MODE) return;
        deformHandlPlace.reset();
        if (multiBody) { multiBody->fullReset(); multiBody->setRigidMode(true); multiBody->initPhysics(); }
        deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
    };

    a.onHandleRadiusChanged = [](float r) {
        gGroupRadius = r;
    };

    a.onSaveAR = []() { saveARimage = true; };

    a.onToggleClusterVis = []() {
        g_showClusterVisualization = !g_showClusterVisualization;
    };

    a.onToggleOrgan = [](int i) {
        if (i < 0 || i >= (int)meshAlphaValues.size()) return;
        float a = meshAlphaValues[i];
        meshAlphaValues[i] = (a < 0.01f) ? 0.5f : (a < 0.75f) ? 1.0f : 0.0f;
    };

    a.onSwitchToDeformMode = []() {
        std::filesystem::create_directories(REG_MODEL_PATH);
        liverMesh3D->exportObjFile(Reg_TARGET_FILE_PATH);
        portalMesh3D->exportObjFile(Reg_PORTAL_FILE_PATH);
        veinMesh3D->exportObjFile(Reg_VEIN_FILE_PATH);
        tumorMesh3D->exportObjFile(Reg_TUMOR_FILE_PATH);
        segmentMesh3D->exportObjFile(Reg_SEGMENT_FILE_PATH);
        gbMesh3D->exportObjFile(Reg_GB_FILE_PATH);
        currentMainMode = DEFORM_MODE;
    };

    a.onResetCamera = []() {
        OrbitCam.resetToInitialState();
        std::cout << "Camera reset to initial position" << std::endl;
    };

    a.onStartFromDepth = []() {
        std::cout << "\n=== RESTART FROM DEPTH ===" << std::endl;
        showProgressOverlay(0.02f, "Reloading meshes...");

        if (multiBody) { delete multiBody; multiBody = nullptr; }
        deformInit = false;
        deformHandlPlace.reset();

        registrationHandle.reset();
        g_showClusterVisualization = false;
        g_showCorrespondencePoints = false;

        auto reloadMesh = [](mCutMesh*& mesh, const std::string& path, glm::vec3 color) {
            if (mesh) { mesh->cleanup(); delete mesh; mesh = nullptr; }
            mCutMesh loader;
            mesh = new mCutMesh(loader.loadMeshFromFile(path.c_str()));
            mesh->mColor = color;
            setUp(*mesh);
        };
        normalizeAndSavePreReg();
        reloadMesh(liverMesh3D,   PreReg_TARGET_FILE_PATH,  glm::vec3(0.8f, 0.2f, 0.2f));
        reloadMesh(portalMesh3D,  PreReg_PORTAL_FILE_PATH,  glm::vec3(0.2f, 0.2f, 0.8f));
        reloadMesh(veinMesh3D,    PreReg_VEIN_FILE_PATH,    glm::vec3(0.2f, 0.5f, 0.5f));
        reloadMesh(tumorMesh3D,   PreReg_TUMOR_FILE_PATH,   glm::vec3(0.8f, 0.5f, 0.5f));
        reloadMesh(segmentMesh3D, PreReg_SEGMENT_FILE_PATH, glm::vec3(0.2f, 0.8f, 0.5f));
        reloadMesh(gbMesh3D,      PreReg_GB_FILE_PATH,      glm::vec3(0.2f, 0.8f, 0.2f));

        allMeshes.clear();
        allMeshes.push_back(liverMesh3D);
        allMeshes.push_back(portalMesh3D);
        allMeshes.push_back(veinMesh3D);
        allMeshes.push_back(tumorMesh3D);
        allMeshes.push_back(segmentMesh3D);
        allMeshes.push_back(gbMesh3D);

        showProgressOverlay(0.05f, "Running depth inference...");
        gCameraPreview.stop();
        gCameraPreview.clearFrozen();
        if (screenMesh) { screenMesh->cleanup(); delete screenMesh; screenMesh = nullptr; }
        initScreenMeshWithDepthRunner(gDepthRunner, screenMesh, false, showProgressOverlay);

        currentMainMode = REGISTRATION_MODE;
        splitScreenMode = false;
        depthSplitScreenMode = false;

        OrbitCam.cx = gWindowWidth / 2.0f;
        OrbitCam.cy = gWindowHeight / 2.0f;
        OrbitCam.gRadius = OrbitCam.InitialRadius;

        gUserSegPoints.clear();
        gUserSegPoints3D.clear();
        gUserSegPointsFG.clear();
        gDroppedFilePath = "";

        meshAlphaValues = {0.8f, 0.9f, 0.9f, 0.9f, 0.5f, 0.5f, 0.7f};

        gUIManager.resetToDepthPhase();

        showProgressOverlay(1.0f, "Restart complete!");
        std::cout << "=== Restart complete - back to Depth phase ===\n" << std::endl;
    };
}

static void computeIdealVoxelSizes() {
    if (!screenMesh || screenMesh->depthImageData.empty()) {
        g_idealVoxel1to1 = g_idealVoxel1to15 = g_idealVoxel1to2 = 0.0f;
        return;
    }
    if (g_refineVertexIndices.empty()) {
        g_idealVoxel1to1 = g_idealVoxel1to15 = g_idealVoxel1to2 = 0.0f;
        return;
    }

    Reg3DCustom::NoOpen3DRegistration reg;
    float zThresh = std::max(0.01f, gDepthScale * 0.05f);
    auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gGridWidth, gGridHeight(), zThresh);
    size_t T = targetCloud->size();
    if (T == 0) { g_idealVoxel1to1 = g_idealVoxel1to15 = g_idealVoxel1to2 = 0.0f; return; }

    auto sourceCloud = std::make_shared<Reg3DCustom::PointCloud>();
    for (size_t idx : g_refineVertexIndices) {
        if (idx * 3 + 2 < liverMesh3D->mVertices.size())
            sourceCloud->addPoint(glm::vec3(liverMesh3D->mVertices[idx*3],
                                            liverMesh3D->mVertices[idx*3+1],
                                            liverMesh3D->mVertices[idx*3+2]));
    }
    if (sourceCloud->empty()) { g_idealVoxel1to1 = g_idealVoxel1to15 = g_idealVoxel1to2 = 0.0f; return; }

    auto findVoxelForRatio = [&](float ratio) -> float {
        float lo = 0.01f, hi = 2.0f;
        for (int i = 0; i < 20; i++) {
            float mid = (lo + hi) * 0.5f;
            auto down = reg.voxelDownSample(sourceCloud, mid);
            float r = (T > 0) ? (float)down->size() / (float)T : 0.0f;
            if (r > ratio) lo = mid; else hi = mid;
        }
        return (lo + hi) * 0.5f;
    };

    g_idealVoxel1to1  = findVoxelForRatio(1.0f);
    g_idealVoxel1to15 = findVoxelForRatio(1.0f / 1.5f);
    g_idealVoxel1to2  = findVoxelForRatio(0.5f);
}

void syncUIState() {
    auto& s = gUIManager.state;
    s.mainMode = (currentMainMode == REGISTRATION_MODE) ? 0 : 1;

    if (gCameraPreview.frozen)      s.cameraState = 2;
    else if (gCameraPreview.active) s.cameraState = 1;
    else                            s.cameraState = 0;

    s.depthScale = gDepthScale;
    s.depthDone = screenMesh && !screenMesh->depthImageData.empty();

    int fg = 0, bg = 0;
    for (const auto& p : gUserSegPoints)
        if (p.isForeground) fg++; else bg++;
    s.segFG = fg;
    s.segBG = bg;

    s.hasLocalImage = !gDroppedFilePath.empty() && gCameraPreview.frozen;
    s.localImageName = gDroppedFilePath;

    switch (registrationHandle.state) {
    case RegistrationData::IDLE:                    s.regState = 0; break;
    case RegistrationData::SELECTING_BOARD_POINTS:  s.regState = 1; break;
    case RegistrationData::SELECTING_OBJECT_POINTS: s.regState = 2; break;
    case RegistrationData::READY_TO_REGISTER:       s.regState = 3; break;
    case RegistrationData::REGISTERED:              s.regState = 4; break;
    case RegistrationData::REFINING:                s.regState = 5; break;
    }
    s.refineEnabled = (registrationHandle.state == RegistrationData::REGISTERED &&
                       !g_refineVertexIndices.empty());
    s.poseLibraryOpen = g_poseLibrary.showWindow;
    s.poseUndoAvailable = g_poseLibrary.hasLastRegistration;
    s.poseEntryCount = (int)g_poseLibrary.entries.size();
    s.boardPtCount  = registrationHandle.boardPoints.size();
    s.objPtCount    = registrationHandle.objectPoints.size();
    s.targetPtCount = registrationHandle.targetPointCount;
    s.splitScreen   = splitScreenMode;
    s.depthSplitScreen = depthSplitScreenMode;
    s.useRegistration = registrationHandle.useRegistration;

    if (registrationHandle.useRegistration) {
        const float* m = glm::value_ptr(registrationHandle.registrationMatrix);
        for (int i = 0; i < 16; i++) s.regMatrix[i] = m[i];
        s.avgError = registrationHandle.compAvgError;
        s.rmse = registrationHandle.compRmse;
        s.maxError = registrationHandle.compMaxError;
        s.scaleFactor = registrationHandle.scaleFactor;
    }

    if (liverMesh3D && !liverMesh3D->mVertices.empty()) {
        glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
        for (size_t i = 0; i < liverMesh3D->mVertices.size(); i += 3) {
            glm::vec3 v(liverMesh3D->mVertices[i], liverMesh3D->mVertices[i+1], liverMesh3D->mVertices[i+2]);
            bmin = glm::min(bmin, v);
            bmax = glm::max(bmax, v);
        }
        s.modelBBoxDiag = glm::length(bmax - bmin);
    }

    switch (deformHandlPlace.state) {
    case DeformHandlPlaceData::RIGID_MODE:        s.deformState = 0; break;
    case DeformHandlPlaceData::HANDLE_PLACE_MODE: s.deformState = 1; break;
    case DeformHandlPlaceData::DEFORM_MODE:       s.deformState = 2; break;
    case DeformHandlPlaceData::PLANECUT_MODE:     s.deformState = 3; break;
    }
    s.handleGroups = multiBody ? (int)multiBody->handleGroups.size() : 0;
    s.maxHandleGroups = SoftBody::MAX_HANDLE_GROUPS;
    s.handleRadius = gGroupRadius;

    for (int i = 0; i < 6; i++)
        s.organs[i].alpha = meshAlphaValues[i];

    s.boardAlpha = meshAlphaValues[6];

    s.clusterVis = g_showClusterVisualization;
    s.correspondenceVis = g_showCorrespondencePoints;
    s.hemiVoxelSize = g_voxelSize;
    s.idealVoxel1to1  = g_idealVoxel1to1;
    s.idealVoxel1to15 = g_idealVoxel1to15;
    s.idealVoxel1to2  = g_idealVoxel1to2;

    s.depthModelIdx = gCurrentDepthModel;
    for (int i = 0; i < DEPTH_MODEL_COUNT; i++)
        s.depthModelAvail[i] = isDepthModelAvailable(i);
}

int main()
{
    initPaths();
    initFilePaths();
    normalizeAndSavePreReg();
    initDepthRunnerConfig(gDepthRunner);

    if (!initOpenGL())
    {
        std::cerr << "GLFW initialization failed" << std::endl;
        return -1;
    }

    g_progressCallback = showProgressOverlay;

    OrbitCam.setWindowSizePointers(&gWindowWidth, &gWindowHeight);
    OrbitCam.setGlobalMatrixPointers(&view, &projection, &model, &objPos);

    ShaderProgram shaderProgram;
    ShaderProgram shaderProgramCube;

    shaderProgram.loadShaders((SHADERS_PATH + "basic.vert").c_str(),
                              (SHADERS_PATH + "basic.frag").c_str());
    shaderProgramCube.loadShaders((SHADERS_PATH + "texture.vert").c_str(),
                                  (SHADERS_PATH + "texture.frag").c_str());

    /* Ctrl+Shift+E 軌道検索のスクリーンショット保存で参照するため、
       シェーダへのポインタをグローバルに公開 (key handler からアクセス可能に)。 */
    g_shaderProgram     = &shaderProgram;
    g_shaderProgramCube = &shaderProgramCube;

    OrbitCam.setIntrinsics(800.0f, 800.0f, gWindowWidth/2.0f, gWindowHeight/2.0f);

    OrbitCam.printCameraInfo();

    deformSphereMarker.generate(1.0f, 16, 16);
    deformSphereMarker.setup();

    liverMesh3D = new mCutMesh(liverMesh3D->loadMeshFromFile(PreReg_TARGET_FILE_PATH.c_str()));
    liverMesh3D->mColor = glm::vec3(0.8f, 0.2f, 0.2f);
    setUp(*liverMesh3D);
    portalMesh3D = new mCutMesh(portalMesh3D->loadMeshFromFile(PreReg_PORTAL_FILE_PATH.c_str()));
    portalMesh3D->mColor = glm::vec3(0.2f, 0.2f, 0.8f);
    setUp(*portalMesh3D);
    veinMesh3D = new mCutMesh(veinMesh3D->loadMeshFromFile(PreReg_VEIN_FILE_PATH.c_str()));
    veinMesh3D->mColor = glm::vec3(0.2f, 0.5f, 0.5f);
    setUp(*veinMesh3D);
    tumorMesh3D = new mCutMesh(tumorMesh3D->loadMeshFromFile(PreReg_TUMOR_FILE_PATH.c_str()));
    tumorMesh3D->mColor = glm::vec3(0.8f, 0.5f, 0.5f);
    setUp(*tumorMesh3D);
    segmentMesh3D = new mCutMesh(segmentMesh3D->loadMeshFromFile(PreReg_SEGMENT_FILE_PATH.c_str()));
    segmentMesh3D->mColor = glm::vec3(0.2f, 0.8f, 0.5f);
    setUp(*segmentMesh3D);
    gbMesh3D = new mCutMesh(gbMesh3D->loadMeshFromFile(PreReg_GB_FILE_PATH.c_str()));
    gbMesh3D->mColor = glm::vec3(0.2f, 0.8f, 0.2f);
    setUp(*gbMesh3D);

    allMeshes.push_back(liverMesh3D);
    allMeshes.push_back(portalMesh3D);
    allMeshes.push_back(veinMesh3D);
    allMeshes.push_back(tumorMesh3D);
    allMeshes.push_back(segmentMesh3D);
    allMeshes.push_back(gbMesh3D);

    snapshotInitialPose();

    initScreenMeshWithDepthRunner(gDepthRunner, screenMesh, cameraUse);

    registrationSphereMarker.generate(1.0f, 16, 16);
    registrationSphereMarker.setup();

    float dt = 1.0f / 60.0f;
    glm::vec3 gravity(0.0f, 0.0f, 0.0f);

    Grabber grabber;
    gGrabber = &grabber;

    setupUICallbacks();

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(gWindow))
    {
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;

        showFPS(gWindow);

        glfwPollEvents();

        if (gFileDropped) {
            gFileDropped = false;
            clearSegPoints();
            gCameraPreview.loadLocalImageAsFrozen(screenMesh, gDroppedFilePath);
            depthSplitScreenMode = true;
            splitScreenMode = true;
            OrbitCamLeft_Target = OrbitCam;
            OrbitCamRight_Screen = OrbitCam;
            OrbitCamLeft_Target.currentTarget = TARGET_LIVER;
            OrbitCamLeft_Target.cx = (gWindowWidth / 2) / 2.0f;
            OrbitCamLeft_Target.cy = gWindowHeight / 2.0f;
            OrbitCamRight_Screen.currentTarget = TARGET_TEXTURE;
            OrbitCamRight_Screen.gRadius = OrbitCam.InitialRadius * 2.0f;
            OrbitCamRight_Screen.cx = (gWindowWidth / 2) / 2.0f;
            OrbitCamRight_Screen.cy = gWindowHeight / 2.0f;
            registrationHandle.reset();
            registrationHandle.state = RegistrationData::IDLE;
            g_refineVertexIndices.clear();
            g_cluster1Points.clear();
            g_cluster2Points.clear();
            g_targetPoints.clear();
            g_showClusterVisualization = false;
            g_showCorrespondencePoints = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        syncUIState();

        if (registrationHandle.state == RegistrationData::REFINING && g_refineState.active) {
            auto stepResult = NormalRefine::refineStep(g_refineState, OrbitCam.cameraDirection);
            const char* mtag = NormalRefine::methodTag(g_refineState.method);

            if (stepResult.correspondenceCount >= 6 && !stepResult.converged) {
                // Apply this step's transform to meshes and accumulate
                NormalRefine::applyIncrementalTransform(stepResult.incrementalTransform,
                                                        g_refineState.organMeshes);

                // Track cumulative transform
                g_refineState.cumulativeTransform =
                    stepResult.incrementalTransform * g_refineState.cumulativeTransform;

                // Evaluate using unified Target→Source metric (measures actual visual fit)
                computeUnifiedMetrics();
                float unifiedRmse = registrationHandle.compRmse;

                if (unifiedRmse < g_refineState.bestRMSE) {
                    g_refineState.bestRMSE = unifiedRmse;
                    g_refineState.bestCumulativeTransform = g_refineState.cumulativeTransform;
                    g_refineState.bestIteration = g_refineState.totalIterations;
                    g_refineState.worseCount = 0;
                } else {
                    g_refineState.worseCount++;
                }

                // Early stop if RMSE worsening for too long
                if (g_refineState.worseCount >= 30) {
                    stepResult.converged = true;
                    std::cout << mtag << " Early stop: unified RMSE worsening for 30 iterations" << std::endl;
                }

                if (g_refineState.totalIterations % 10 == 0) {
                    std::cout << mtag << " iter=" << g_refineState.totalIterations
                              << " corr=" << stepResult.correspondenceCount
                              << " internal=" << std::fixed << std::setprecision(4)
                              << stepResult.rmse
                              << " unified=" << unifiedRmse
                              << " best=" << g_refineState.bestRMSE
                              << "@" << g_refineState.bestIteration << std::endl;
                }
            }

            if (stepResult.converged) {
                g_refineState.active = false;
                registrationHandle.state = RegistrationData::REGISTERED;
                const char* mname = NormalRefine::methodName(g_refineState.method);

                // Revert to initial, then apply best transform
                bool improved = g_refineState.bestRMSE < g_refineState.initialRMSE;
                g_refineState.restoreMeshes();
                if (improved) {
                    NormalRefine::applyIncrementalTransform(
                        g_refineState.bestCumulativeTransform,
                        g_refineState.organMeshes);
                }

                std::cout << "\n=== " << mname << " CONVERGED ===" << std::endl;
                std::cout << "  Iterations: " << g_refineState.totalIterations << std::endl;
                std::cout << "  Initial RMSE: " << std::fixed << std::setprecision(4)
                          << g_refineState.initialRMSE << std::endl;
                std::cout << "  Best RMSE:    " << g_refineState.bestRMSE
                          << " (iter " << g_refineState.bestIteration << ")" << std::endl;
                if (improved) {
                    std::cout << "  Improvement:  " << std::setprecision(1)
                    << (1.0f - g_refineState.bestRMSE / g_refineState.initialRMSE) * 100.0f
                    << "%" << std::endl;
                    std::cout << "  >> Reverted to best state (iter "
                              << g_refineState.bestIteration << ")" << std::endl;
                } else {
                    std::cout << "  >> No improvement — reverted to initial state" << std::endl;
                }
                // Restore cout precision (setprecision(1) above contaminates subsequent output)
                std::cout << std::defaultfloat << std::setprecision(6);

                // Update refine metrics and unified comparison (same as manual stop)
                registrationHandle.refineCount++;
                registrationHandle.refineInitialRMSE   = g_refineState.initialRMSE;
                registrationHandle.refineBestRMSE      = g_refineState.bestRMSE;
                registrationHandle.refineBestIteration  = g_refineState.bestIteration;
                computeUnifiedMetrics();
                poseSaveToLibrary();
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        OrbitCam.UpdateCamera(deltaTime);

        if (cameraUse && screenMesh->isUsingCamera()) {
            screenMesh->updateTextureFromCamera();
        }
        gCameraPreview.update(screenMesh);

        if(currentMainMode == REGISTRATION_MODE) {

            if (depthSplitScreenMode) {
                glm::vec3 liverCenter = OrbitCamLeft_Target.calculateMeshCenter(liverMesh3D->mVertices);
                glm::vec3 textureCenter = OrbitCamRight_Screen.calculateMeshCenter(screenMesh->mVertices);
                OrbitCamLeft_Target.updateTargetPositions(liverCenter, glm::vec3(0));
                OrbitCamRight_Screen.updateTargetPositions(glm::vec3(0), textureCenter);
                renderDepthSplitScreen(shaderProgram, shaderProgramCube);

            } else if (!splitScreenMode) {
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                glm::vec3 liverCenter = OrbitCam.calculateMeshCenter(liverMesh3D->mVertices);
                glm::vec3 textureCenter = OrbitCam.calculateMeshCenter(screenMesh->mVertices);
                OrbitCam.updateTargetPositions(liverCenter, textureCenter);
                model = glm::translate(glm::mat4(1.0f), objPos);

                std::vector<glm::vec4> customColors = {
                    glm::vec4(0.8f, 0.2f, 0.2f, meshAlphaValues[0]),
                    glm::vec4(0.9f, 0.6f, 0.6f, meshAlphaValues[1]),
                    glm::vec4(0.2f, 0.8f, 0.8f, meshAlphaValues[2]),
                    glm::vec4(0.8f, 0.2f, 0.8f, meshAlphaValues[3]),
                    glm::vec4(0.8f, 0.8f, 0.0f, meshAlphaValues[4]),
                    glm::vec4(0.2f, 0.5f, 0.2f, meshAlphaValues[5]),
                    glm::vec4(1.0f, 1.0f, 1.0f, meshAlphaValues[6])
                };

                std::vector<mCutMesh*> meshesToDraw = {
                    liverMesh3D, portalMesh3D, veinMesh3D,
                    tumorMesh3D, segmentMesh3D, gbMesh3D, screenMesh
                };

                draw_AllmCutMeshes(meshesToDraw, shaderProgram, shaderProgramCube,
                                   OrbitCam.cameraPos, customColors,
                                   model, view, projection, 6);

                if (g_showClusterVisualization) {
                    // std::cout << "Drawing clusters: " << g_cluster1Points.size()
                    // << " + " << g_cluster2Points.size()
                    // << " + " << g_targetPoints.size() << std::endl;

                    for (size_t i = 0; i < g_cluster1Points.size(); i++) {
                        registrationSphereMarker.draw(shaderProgram, g_cluster1Points[i],
                                                      glm::vec3(0.0f, 1.0f, 0.0f),
                                                      0.08f, view, projection, OrbitCam.cameraPos);
                    }

                    for (size_t i = 0; i < g_cluster2Points.size(); i++) {
                        registrationSphereMarker.draw(shaderProgram, g_cluster2Points[i],
                                                      glm::vec3(0.0f, 0.5f, 1.0f),
                                                      0.08f, view, projection, OrbitCam.cameraPos);
                    }

                    for (size_t i = 0; i < g_targetPoints.size(); i++) {
                        registrationSphereMarker.draw(shaderProgram, g_targetPoints[i],
                                                      glm::vec3(1.0f, 1.0f, 0.0f),
                                                      0.12f, view, projection, OrbitCam.cameraPos);
                    }
                }

                {
                    bool activeSelection = (registrationHandle.state == RegistrationData::SELECTING_BOARD_POINTS ||
                                            registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS ||
                                            registrationHandle.state == RegistrationData::READY_TO_REGISTER);
                    if (activeSelection || g_showCorrespondencePoints) {
                        for (size_t i = 0; i < registrationHandle.boardPoints.size(); i++) {
                            glm::vec3 color = getPointColor(i, true);
                            registrationSphereMarker.draw(shaderProgram, registrationHandle.boardPoints[i],
                                                          color, 0.3f, view, projection, OrbitCam.cameraPos);
                        }
                        for (size_t i = 0; i < registrationHandle.objectPoints.size(); i++) {
                            glm::vec3 color = getPointColor(i, false);
                            registrationSphereMarker.draw(shaderProgram, registrationHandle.objectPoints[i],
                                                          color, 0.3f, view, projection, OrbitCam.cameraPos);
                        }
                    }
                }

                if (!gUserSegPoints3D.empty()) {
                    for (size_t i = 0; i < gUserSegPoints3D.size(); i++) {
                        glm::vec3 color = gUserSegPointsFG[i]
                                              ? glm::vec3(0.0f, 1.0f, 0.0f)
                                              : glm::vec3(1.0f, 0.0f, 0.0f);
                        registrationSphereMarker.draw(
                            shaderProgram, gUserSegPoints3D[i],
                            color, 0.15f, view, projection, OrbitCam.cameraPos);
                    }
                }

            } else {

                glm::vec3 liverCenter = OrbitCamLeft_Target.calculateMeshCenter(liverMesh3D->mVertices);
                glm::vec3 textureCenter = OrbitCamRight_Screen.calculateMeshCenter(screenMesh->mVertices);
                OrbitCamLeft_Target.updateTargetPositions(liverCenter, glm::vec3(0));
                OrbitCamRight_Screen.updateTargetPositions(glm::vec3(0), textureCenter);
                renderSplitScreen(shaderProgram, shaderProgramCube);
            }

        }

        if(currentMainMode == DEFORM_MODE) {
            if(!deformInit){

                CentVoxTetrahedralizerHybrid tetrahedralizer(
                    20, Reg_TARGET_FILE_PATH, OUTPUT_TET_FILE,
                    CentVoxTetrahedralizerHybrid::DetectionMode::HYBRID, 1, 1);
                CentVoxTetrahedralizerHybrid::SmoothingSettings ss;
                ss.enabled=false; ss.iterations=0; ss.smoothFactor=0.0f; ss.preserveVolume=true; ss.rescaleToOriginal=true;
                tetrahedralizer.setSmoothingSettings(ss);
                tetrahedralizer.execute();


                // VoxelTetrahedralizer tetrahedralizer(DEFAULT_GRID_SIZE, Reg_TARGET_FILE_PATH, OUTPUT_TET_FILE);

                // tetrahedralizer.setSmoothingEnabled(true, SMOOTH_ITERATION, SMOOTH_FACTOR, true);

                // VoxelTetrahedralizer::InflationSettings inflationSettings;
                // if(DEFAULT_GRID_SIZE<30){
                //     inflationSettings.enabled = true;} else {inflationSettings.enabled = false;};
                // inflationSettings.targetCoverage = 99.0f;
                // inflationSettings.successThreshold = 99.0f;
                // tetrahedralizer.setInflationSettings(inflationSettings);

                // MeshDataTypes::SimpleMeshData resultData = tetrahedralizer.execute();

                std::cout << "\n=== Tetrahedral mesh generated ===" << std::endl;
                std::cout << "Output file: " << OUTPUT_TET_FILE << std::endl;
                std::cout << "Smoothing: Enabled" << std::endl;

                SoftBody::MeshData liver_mesh = TetoMeshData::ReadVetexAndFace(Reg_TARGET_FILE_PATH);
                SoftBody::MeshData tetmesh = SoftBody::loadTetMesh(OUTPUT_TET_FILE);
                SoftBody::MeshData portal_mesh = TetoMeshData::ReadVetexAndFace(Reg_PORTAL_FILE_PATH);
                SoftBody::MeshData vein_mesh = TetoMeshData::ReadVetexAndFace(Reg_VEIN_FILE_PATH);
                SoftBody::MeshData tumor_mesh = TetoMeshData::ReadVetexAndFace(Reg_TUMOR_FILE_PATH);
                SoftBody::MeshData res_mesh = TetoMeshData::ReadVetexAndFace(Reg_SEGMENT_FILE_PATH);
                SoftBody::MeshData gb_mesh = TetoMeshData::ReadVetexAndFace(Reg_GB_FILE_PATH);

                std::vector<SoftBody::MeshData> visMeshes;
                visMeshes.push_back(liver_mesh);
                visMeshes.push_back(portal_mesh);
                visMeshes.push_back(vein_mesh);
                visMeshes.push_back(tumor_mesh);
                visMeshes.push_back(res_mesh);
                visMeshes.push_back(gb_mesh);

                multiBody = new SoftBody(tetmesh, visMeshes,0.001f, 0.0f);

                gGrabber->setPhysicsObject(multiBody);
                multiBody->setRigidMode(true);
                deformInit = true;
            }

            gGrabber->update(dt);
            model = glm::translate(glm::mat4(1.0f), bunnyPos);
            multiBody->setModelMatrix(model);

            if (deformHandlPlace.state != DeformHandlPlaceData::PLANECUT_MODE)
                for (size_t g = 0; g < multiBody->handleGroups.size(); g++) {
                    glm::vec3 center = multiBody->handleGroups[g].centerPosition;
                    float radius = multiBody->handleGroups[g].radius;
                    glm::vec3 worldPos = glm::vec3(model * glm::vec4(center, 1.0f));
                    glm::vec3 color = getPointColor(g, true);
                    deformSphereMarker.draw(shaderProgram, worldPos, color, radius,
                                            view, projection, OrbitCam.cameraPos);
                }

            shaderProgram.use();
            shaderProgram.setUniform("model", model);
            shaderProgram.setUniform("lightPos", OrbitCam.cameraPos);
            shaderProgram.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
            shaderProgram.setUniform("view", view);
            shaderProgram.setUniform("projection", projection);
            shaderProgram.setUniform("vertColor", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

            int numSubsteps = 1;
            float stepDt = dt / float(numSubsteps);
            for (int i = 0; i < numSubsteps; i++) {
                multiBody->preSolve(stepDt, gravity);
                multiBody->solve(stepDt);
                multiBody->postSolve(stepDt);
            }

            multiBody->updateVisMeshes();

            if (gPendingRmseFrames > 0) {
                gPendingRmseFrames--;
                if (gPendingRmseFrames == 0 && currentMainMode == DEFORM_MODE && screenMesh) {
                    const auto& visV = multiBody->getVisPositions(0);
                    float settled = AutoDeform::measureRMSE(
                        gAutoDeform, visV, screenMesh,
                        gGridWidth, gGridHeight(), gDepthScale,
                        /*storeAsBefore=*/false, 1.0f, /*verbose=*/false);
                    float base = gAutoDeform.rmseBeforeDeform;
                    float stepDelta = settled - gPendingRmseImmediate;
                    float baseDelta = (base > 0.0f) ? (settled - base) : 0.0f;
                    std::cout << "[settled] RMSE=" << settled
                              << "  step:" << (stepDelta>=0?"+":"") << stepDelta << (stepDelta<0?" DOWN":" UP")
                              << "  base:" << (baseDelta>=0?"+":"") << baseDelta << (baseDelta<0?" DOWN":" UP")
                              << std::endl;
                }
            }

            if (gAutoDeformDebugFrames > 0) {
                const auto& cur = multiBody->getPositions();
                float maxD = 0.0f, sumD = 0.0f; int cnt = 0;
                if (!gAutoDeformDebugBaseline.empty() && cur.size() == gAutoDeformDebugBaseline.size()) {
                    for (size_t k = 0; k + 2 < cur.size(); k += 3) {
                        glm::vec3 a(cur[k], cur[k+1], cur[k+2]);
                        glm::vec3 b(gAutoDeformDebugBaseline[k], gAutoDeformDebugBaseline[k+1], gAutoDeformDebugBaseline[k+2]);
                        float d = glm::length(a - b);
                        if (d > maxD) maxD = d;
                        sumD += d; cnt++;
                    }
                }
                std::cout << "[DBG f=" << gAutoDeformDebugFrames << "] rigid=" << multiBody->isRigidMode()
                          << " att=" << multiBody->getNumAttachments()
                          << " maxDisp=" << maxD
                          << " avgDisp=" << (cnt > 0 ? sumD / cnt : 0.0f)
                          << " sub=" << (int)deformHandlPlace.state << std::endl;
                gAutoDeformDebugFrames--;
            }

            std::vector<glm::vec4> customColors = {
                glm::vec4(0.8f, 0.2f, 0.2f, meshAlphaValues[0]),
                glm::vec4(0.9f, 0.6f, 0.6f, meshAlphaValues[1]),
                glm::vec4(0.2f, 0.8f, 0.8f, meshAlphaValues[2]),
                glm::vec4(0.8f, 0.2f, 0.8f, meshAlphaValues[3]),
                glm::vec4(0.8f, 0.8f, 0.0f, meshAlphaValues[4]),
                glm::vec4(0.2f, 0.5f, 0.2f, meshAlphaValues[5]),
            };

            glm::vec4 screenMeshColor = glm::vec4(1.0f, 1.0f, 1.0f, meshAlphaValues[6]);

            draw_AllVisMeshesWithExtraMesh(
                multiBody, shaderProgram, shaderProgramCube,
                screenMesh, OrbitCam.cameraPos,
                customColors, screenMeshColor,
                model, view, projection);


            // draw_AllVisMeshesWithExtraMesh(...) の直後に追加
            multiBody->updateTetMeshes();
            shaderProgram.setUniform("model", model);
            multiBody->drawTetMesh(shaderProgram);

            AutoDeform::draw(gAutoDeform, deformSphereMarker, shaderProgram,
                             view, projection, OrbitCam.cameraPos);

            if (gAutoCtrl.numFix() + gAutoCtrl.numMove() > 0) {
                const auto& presetCur = AutoDeform::getPresets()[gAutoDeformPresetIdx];
                float fixRadiusUnit  = gAutoDeform.fieldBboxDiag * presetCur.rFixScale  * 0.5f;
                float moveRadiusUnit = gAutoDeform.fieldBboxDiag * presetCur.rMoveScale * 0.5f;
                drawAutoHandles(
                    gAutoCtrl, deformSphereMarker, shaderProgram,
                    view, projection, OrbitCam.cameraPos,
                    fixRadiusUnit, moveRadiusUnit);
            }
        }

        if (saveARimage) {
            saveARimage = false;

            const int AR_W = 1280;
            const int AR_H = 720;

            glm::mat4 savedView = view, savedProj = projection, savedModel = model;

            GLuint fbo, colorTex, depthRbo;
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            glGenTextures(1, &colorTex);
            glBindTexture(GL_TEXTURE_2D, colorTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, AR_W, AR_H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

            glGenRenderbuffers(1, &depthRbo);
            glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, AR_W, AR_H);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRbo);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                FullSphereCamera arCam = OrbitCam;
                arCam.resetToInitialState();
                arCam.useIntrinsics = false;
                arCam.cx = AR_W / 2.0f;
                arCam.cy = AR_H / 2.0f;
                float boardScale = 10.0f;
                float halfFOVy = atan(AR_H / (2.0f * arCam.fy));
                arCam.gRadius = (boardScale / 2.0f) / tan(halfFOVy);
                arCam.gFOV = glm::degrees(2.0f * halfFOVy);
                arCam.currentTarget = TARGET_TEXTURE;

                glm::vec3 textureCenter = arCam.calculateMeshCenter(screenMesh->mVertices);
                arCam.updateTargetPositions(glm::vec3(0), textureCenter);
                arCam.UpdateCamera();

                glViewport(0, 0, AR_W, AR_H);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                model = glm::translate(glm::mat4(1.0f),
                                       (currentMainMode == DEFORM_MODE) ? bunnyPos : objPos);

                if (currentMainMode == DEFORM_MODE && multiBody != nullptr) {
                    shaderProgram.use();
                    shaderProgram.setUniform("model", model);
                    shaderProgram.setUniform("lightPos", arCam.cameraPos);
                    shaderProgram.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
                    shaderProgram.setUniform("view", arCam.view);
                    shaderProgram.setUniform("projection", arCam.projection);

                    std::vector<glm::vec4> customColors = {
                        glm::vec4(0.8f, 0.2f, 0.2f, meshAlphaValues[0]),
                        glm::vec4(0.9f, 0.6f, 0.6f, meshAlphaValues[1]),
                        glm::vec4(0.2f, 0.8f, 0.8f, meshAlphaValues[2]),
                        glm::vec4(0.8f, 0.2f, 0.8f, meshAlphaValues[3]),
                        glm::vec4(0.8f, 0.8f, 0.0f, meshAlphaValues[4]),
                        glm::vec4(0.2f, 0.5f, 0.2f, meshAlphaValues[5]),
                    };
                    glm::vec4 screenMeshColor = glm::vec4(1.0f, 1.0f, 1.0f, meshAlphaValues[6]);

                    draw_AllVisMeshesWithExtraMesh(
                        multiBody, shaderProgram, shaderProgramCube,
                        screenMesh, arCam.cameraPos,
                        customColors, screenMeshColor,
                        model, arCam.view, arCam.projection);
                } else {
                    std::vector<glm::vec4> arColors = {
                        glm::vec4(0.8f, 0.2f, 0.2f, meshAlphaValues[0]),
                        glm::vec4(0.9f, 0.6f, 0.6f, meshAlphaValues[1]),
                        glm::vec4(0.2f, 0.8f, 0.8f, meshAlphaValues[2]),
                        glm::vec4(0.8f, 0.2f, 0.8f, meshAlphaValues[3]),
                        glm::vec4(0.8f, 0.8f, 0.0f, meshAlphaValues[4]),
                        glm::vec4(0.2f, 0.5f, 0.2f, meshAlphaValues[5]),
                        glm::vec4(1.0f, 1.0f, 1.0f, meshAlphaValues[6])
                    };

                    std::vector<mCutMesh*> arMeshes = {
                        liverMesh3D, portalMesh3D, veinMesh3D,
                        tumorMesh3D, segmentMesh3D, gbMesh3D, screenMesh
                    };

                    draw_AllmCutMeshes(arMeshes, shaderProgram, shaderProgramCube,
                                       arCam.cameraPos, arColors,
                                       model, arCam.view, arCam.projection, 6);
                }

                std::vector<unsigned char> pixels(AR_W * AR_H * 3);
                glPixelStorei(GL_PACK_ALIGNMENT, 1);   /* avoid row padding for non-mod-4 widths */
                glReadPixels(0, 0, AR_W, AR_H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

                int stride = AR_W * 3;
                std::vector<unsigned char> flipped(pixels.size());
                for (int y = 0; y < AR_H; y++)
                    memcpy(&flipped[y * stride], &pixels[(AR_H - 1 - y) * stride], stride);

                auto now = std::chrono::system_clock::now();
                auto tt = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch()) % 1000;
                struct tm lt;
#ifdef _WIN32
                localtime_s(&lt, &tt);
#else
                localtime_r(&tt, &lt);
#endif
                char stamp[64];
                snprintf(stamp, sizeof(stamp), "%04d%02d%02d_%02d%02d%02d_%03d",
                         lt.tm_year+1900, lt.tm_mon+1, lt.tm_mday,
                         lt.tm_hour, lt.tm_min, lt.tm_sec, (int)ms.count());

                const char* prefixes[] = {"data/","../data/","../../data/","../../../data/","../../../../data/",nullptr};
                bool saved = false;
                for (int pi = 0; prefixes[pi]; pi++) {
                    if (std::filesystem::is_directory(std::string(prefixes[pi]))) {
                        std::string dir = std::string(prefixes[pi]) + "screenshots/";
                        std::filesystem::create_directories(dir);
                        std::string path = dir + "ar_" + stamp + ".png";
                        stbi_write_png(path.c_str(), AR_W, AR_H, 3, flipped.data(), stride);
                        printf("[AR] Screenshot saved: %s\n", std::filesystem::absolute(path).string().c_str());
                        saved = true;
                        break;
                    }
                }
                if (!saved) {
                    std::string fallback = std::string("ar_") + stamp + ".png";
                    stbi_write_png(fallback.c_str(), AR_W, AR_H, 3, flipped.data(), stride);
                    printf("[AR] Screenshot saved (fallback): %s\n",
                           std::filesystem::absolute(fallback).string().c_str());
                }

                if (g_arPreviewTex == 0) glGenTextures(1, &g_arPreviewTex);
                glBindTexture(GL_TEXTURE_2D, g_arPreviewTex);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, AR_W, AR_H, 0, GL_RGB, GL_UNSIGNED_BYTE, flipped.data());
                glBindTexture(GL_TEXTURE_2D, 0);
                g_arPreviewW = AR_W;
                g_arPreviewH = AR_H;
                g_showARPreview = true;
            } else {
                printf("[AR] FBO creation failed!\n");
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteTextures(1, &colorTex);
            glDeleteRenderbuffers(1, &depthRbo);
            glDeleteFramebuffers(1, &fbo);

            view = savedView; projection = savedProj; model = savedModel;
            glViewport(0, 0, gWindowWidth, gWindowHeight);
        }

        // =========================================================
        //  ORTHO AR screenshot (Shift+A): ortho-shadow silhouette
        //  従来の saveARimage と同じ view / メッシュ / 色で、
        //  projection 行列だけ ortho に差し替えて描画する。
        //  目的: パース成分を取り除き、深度０基準の
        //        スクリーンメッシュ輪郭と肝臓 ortho シルエットを比較。
        // =========================================================
        if (saveARimageOrtho) {
            saveARimageOrtho = false;

            const int AR_W = 1280;
            const int AR_H = 720;

            glm::mat4 savedView = view, savedProj = projection, savedModel = model;

            GLuint fbo, colorTex, depthRbo;
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            glGenTextures(1, &colorTex);
            glBindTexture(GL_TEXTURE_2D, colorTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, AR_W, AR_H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

            glGenRenderbuffers(1, &depthRbo);
            glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, AR_W, AR_H);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRbo);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                FullSphereCamera arCam = OrbitCam;
                arCam.resetToInitialState();
                arCam.useIntrinsics = false;
                arCam.cx = AR_W / 2.0f;
                arCam.cy = AR_H / 2.0f;
                float boardScale = 10.0f;
                float halfFOVy = atan(AR_H / (2.0f * arCam.fy));
                arCam.gRadius = (boardScale / 2.0f) / tan(halfFOVy);
                arCam.gFOV = glm::degrees(2.0f * halfFOVy);
                arCam.currentTarget = TARGET_TEXTURE;

                glm::vec3 textureCenter = arCam.calculateMeshCenter(screenMesh->mVertices);
                arCam.updateTargetPositions(glm::vec3(0), textureCenter);
                arCam.UpdateCamera();

                // ===== perspective 版との唯一の違い =====
                // UpdateCamera が作った perspective を ortho で上書き。
                // 垂直方向を boardScale(=10) に固定することで、
                // perspective 版と「垂直フレーミング」が一致する。
                float halfH  = boardScale * 0.5f;             // 5.0
                float aspect = (float)AR_W / (float)AR_H;     // 16/9
                float halfW  = halfH * aspect;                // 8.888...
                arCam.projection = glm::ortho(-halfW, halfW,
                                              -halfH, halfH,
                                              arCam.nearPlane, arCam.farPlane);
                // =========================================

                glViewport(0, 0, AR_W, AR_H);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                model = glm::translate(glm::mat4(1.0f),
                                       (currentMainMode == DEFORM_MODE) ? bunnyPos : objPos);

                if (currentMainMode == DEFORM_MODE && multiBody != nullptr) {
                    shaderProgram.use();
                    shaderProgram.setUniform("model", model);
                    shaderProgram.setUniform("lightPos", arCam.cameraPos);
                    shaderProgram.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
                    shaderProgram.setUniform("view", arCam.view);
                    shaderProgram.setUniform("projection", arCam.projection);

                    std::vector<glm::vec4> customColors = {
                        glm::vec4(0.8f, 0.2f, 0.2f, meshAlphaValues[0]),
                        glm::vec4(0.9f, 0.6f, 0.6f, meshAlphaValues[1]),
                        glm::vec4(0.2f, 0.8f, 0.8f, meshAlphaValues[2]),
                        glm::vec4(0.8f, 0.2f, 0.8f, meshAlphaValues[3]),
                        glm::vec4(0.8f, 0.8f, 0.0f, meshAlphaValues[4]),
                        glm::vec4(0.2f, 0.5f, 0.2f, meshAlphaValues[5]),
                    };
                    glm::vec4 screenMeshColor = glm::vec4(1.0f, 1.0f, 1.0f, meshAlphaValues[6]);

                    draw_AllVisMeshesWithExtraMesh(
                        multiBody, shaderProgram, shaderProgramCube,
                        screenMesh, arCam.cameraPos,
                        customColors, screenMeshColor,
                        model, arCam.view, arCam.projection);
                } else {
                    std::vector<glm::vec4> arColors = {
                        glm::vec4(0.8f, 0.2f, 0.2f, meshAlphaValues[0]),
                        glm::vec4(0.9f, 0.6f, 0.6f, meshAlphaValues[1]),
                        glm::vec4(0.2f, 0.8f, 0.8f, meshAlphaValues[2]),
                        glm::vec4(0.8f, 0.2f, 0.8f, meshAlphaValues[3]),
                        glm::vec4(0.8f, 0.8f, 0.0f, meshAlphaValues[4]),
                        glm::vec4(0.2f, 0.5f, 0.2f, meshAlphaValues[5]),
                        glm::vec4(1.0f, 1.0f, 1.0f, meshAlphaValues[6])
                    };

                    std::vector<mCutMesh*> arMeshes = {
                        liverMesh3D, portalMesh3D, veinMesh3D,
                        tumorMesh3D, segmentMesh3D, gbMesh3D, screenMesh
                    };

                    draw_AllmCutMeshes(arMeshes, shaderProgram, shaderProgramCube,
                                       arCam.cameraPos, arColors,
                                       model, arCam.view, arCam.projection, 6);
                }

                std::vector<unsigned char> pixels(AR_W * AR_H * 3);
                glPixelStorei(GL_PACK_ALIGNMENT, 1);   /* avoid row padding for non-mod-4 widths */
                glReadPixels(0, 0, AR_W, AR_H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

                int stride = AR_W * 3;
                std::vector<unsigned char> flipped(pixels.size());
                for (int y = 0; y < AR_H; y++)
                    memcpy(&flipped[y * stride], &pixels[(AR_H - 1 - y) * stride], stride);

                auto now = std::chrono::system_clock::now();
                auto tt = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch()) % 1000;
                struct tm lt;
#ifdef _WIN32
                localtime_s(&lt, &tt);
#else
                localtime_r(&tt, &lt);
#endif
                char stamp[64];
                snprintf(stamp, sizeof(stamp), "%04d%02d%02d_%02d%02d%02d_%03d",
                         lt.tm_year+1900, lt.tm_mon+1, lt.tm_mday,
                         lt.tm_hour, lt.tm_min, lt.tm_sec, (int)ms.count());

                const char* prefixes[] = {"data/","../data/","../../data/","../../../data/","../../../../data/",nullptr};
                bool saved = false;
                for (int pi = 0; prefixes[pi]; pi++) {
                    if (std::filesystem::is_directory(std::string(prefixes[pi]))) {
                        std::string dir = std::string(prefixes[pi]) + "screenshots/";
                        std::filesystem::create_directories(dir);
                        std::string path = dir + "ar_ortho_" + stamp + ".png";
                        stbi_write_png(path.c_str(), AR_W, AR_H, 3, flipped.data(), stride);
                        printf("[AR-Ortho] Screenshot saved: %s\n", std::filesystem::absolute(path).string().c_str());
                        saved = true;
                        break;
                    }
                }
                if (!saved) {
                    std::string fallback = std::string("ar_ortho_") + stamp + ".png";
                    stbi_write_png(fallback.c_str(), AR_W, AR_H, 3, flipped.data(), stride);
                    printf("[AR-Ortho] Screenshot saved (fallback): %s\n",
                           std::filesystem::absolute(fallback).string().c_str());
                }

                // プレビュー窓も ortho 版で上書き（従来の g_arPreviewTex を再利用）
                if (g_arPreviewTex == 0) glGenTextures(1, &g_arPreviewTex);
                glBindTexture(GL_TEXTURE_2D, g_arPreviewTex);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, AR_W, AR_H, 0, GL_RGB, GL_UNSIGNED_BYTE, flipped.data());
                glBindTexture(GL_TEXTURE_2D, 0);
                g_arPreviewW = AR_W;
                g_arPreviewH = AR_H;
                g_showARPreview = true;
            } else {
                printf("[AR-Ortho] FBO creation failed!\n");
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteTextures(1, &colorTex);
            glDeleteRenderbuffers(1, &depthRbo);
            glDeleteFramebuffers(1, &fbo);

            view = savedView; projection = savedProj; model = savedModel;
            glViewport(0, 0, gWindowWidth, gWindowHeight);
        }

        glViewport(0, 0, gWindowWidth, gWindowHeight);
        gUIManager.draw(gWindowWidth, gWindowHeight);

        {
            ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_Appearing);
            ImGui::SetNextWindowPos(ImVec2(8, gWindowHeight - 140.0f), ImGuiCond_Appearing);
            if (ImGui::Begin("Projection Mode", nullptr,
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
                int prev = gPinMode;
                ImGui::RadioButton("Relief (classical)", &gPinMode, 0);
                ImGui::RadioButton("Diff Pinhole (recommended)", &gPinMode, 1);
                ImGui::RadioButton("Pure Pinhole (geometric ref)", &gPinMode, 2);
                if (gPinMode != 0 && !gLatestIntrinsics.valid) {
                    updateLatestIntrinsicsFromOutputDir();
                }
                if (gPinMode != prev) {
                    if (gPinMode == 1) {
                        gPinholeBaseScale = gDepthScale;
                        std::cout << "[Pinhole] DIFF, baseScale=" << gPinholeBaseScale << std::endl;
                    } else if (gPinMode == 2) {
                        gPinholeBaseScale = 0.0f;
                        std::cout << "[Pinhole] PURE" << std::endl;
                    } else {
                        gPinholeBaseScale = 0.0f;
                        std::cout << "[Pinhole] RELIEF" << std::endl;
                    }
                    if (screenMesh && !screenMesh->depthImageData.empty()) {
                        regenerateDepthMeshAuto(screenMesh, gDepthScale, gMeshScale);
                    }
                }
                if (gLatestIntrinsics.valid) {
                    ImGui::Text("fx=%.0f fy=%.0f", gLatestIntrinsics.fx, gLatestIntrinsics.fy);
                } else if (gPinMode != 0) {
                    ImGui::TextColored(ImVec4(1,0.5f,0,1), "no intrinsics");
                }
                if (gPinMode == 1) {
                    ImGui::Text("base=%.2f  delta=%.2f",
                                gPinholeBaseScale, gDepthScale - gPinholeBaseScale);
                }
            }
            ImGui::End();
        }

        if (g_showARPreview && g_arPreviewTex != 0) {
            float vpW = gUIManager.getViewportWidth(gWindowWidth);
            float prevW = vpW * 0.45f;
            float prevH = prevW * (float)g_arPreviewH / (float)g_arPreviewW;
            float maxH = gWindowHeight * 0.5f;
            if (prevH > maxH) { prevH = maxH; prevW = prevH * (float)g_arPreviewW / (float)g_arPreviewH; }

            ImGui::SetNextWindowSize(ImVec2(prevW + 16, prevH + 50), ImGuiCond_Appearing);
            ImGui::SetNextWindowPos(
                ImVec2(vpW * 0.5f, gWindowHeight * 0.5f), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.04f, 0.04f, 0.06f, 0.95f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            if (ImGui::Begin("AR Screenshot", &g_showARPreview,
                             ImGuiWindowFlags_NoCollapse)) {
                ImVec2 avail = ImGui::GetContentRegionAvail();
                float imgW = avail.x;
                float imgH = imgW * (float)g_arPreviewH / (float)g_arPreviewW;
                if (imgH > avail.y) { imgH = avail.y; imgW = imgH * (float)g_arPreviewW / (float)g_arPreviewH; }
                float offX = (avail.x - imgW) * 0.5f;
                if (offX > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offX);
                ImGui::Image((ImTextureID)(intptr_t)g_arPreviewTex, ImVec2(imgW, imgH));
            }
            ImGui::End();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
        }

        drawPoseLibraryWindow();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        captureSceneForProgress();

        glfwSwapBuffers(gWindow);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    return 0;
}

bool initOpenGL()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    gWindow = glfwCreateWindow(gWindowWidth, gWindowHeight, "Window", NULL, NULL);
    glfwMakeContextCurrent(gWindow);
    glewExperimental = GL_TRUE;
    glewInit();

    glfwSetKeyCallback(gWindow, glfw_onKey);
    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);
    glfwSetFramebufferSizeCallback(gWindow, glfw_OnFramebufferSize);
    glfwSetCursorPosCallback(gWindow, glfw_onMouseMoveOrbit);
    glfwSetScrollCallback(gWindow, glfw_onMouseScroll);
    glfwSetDropCallback(gWindow, glfw_onFileDrop);

    glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPos(gWindow, gWindowWidth / 2.0, gWindowHeight / 2.0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glViewport(0, 0, gWindowWidth, gWindowHeight);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.075f, 0.094f, 1.0f);

    {
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

    ImGui_ImplGlfw_InitForOpenGL(gWindow, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    {
        auto loadIcon = [](const char* subdir, const char* name) -> unsigned int {
            int w, h, ch;
            const char* prefixes[] = {
                "data/",
                "../data/",
                "../../data/",
                "../../../data/",
                "../../../../data/",
                nullptr
            };
            for (int p = 0; prefixes[p]; p++) {
                char path[512];
                snprintf(path, sizeof(path), "%s%s%s_icon.png", prefixes[p], subdir, name);
                unsigned char* data = stbi_load(path, &w, &h, &ch, 4);
                if (data) {
                    GLuint tex;
                    glGenTextures(1, &tex);
                    glBindTexture(GL_TEXTURE_2D, tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                    stbi_image_free(data);
                    printf("[Icon] Loaded: %s (%dx%d)\n", path, w, h);
                    return tex;
                }
            }
            printf("[Icon] Not found: %s%s_icon.png (tried all paths)\n", subdir, name);
            return 0;
        };
        const char* organNames[] = {"liver","portal","vein","tumor","segment","gb"};
        for (int i = 0; i < 6; i++) {
            gUIManager.state.organIconTex[i] = loadIcon("icons/", organNames[i]);
        }
        gUIManager.state.boardIconTex = loadIcon("icons/", "board");
        const char* btnNames[] = {"camera","load_images","depth","full_auto","hemi_auto","umeyama","rigid","handle","deform"};
        for (int i = 0; i < RegUIState::ICON_COUNT; i++) {
            gUIManager.state.btnIconTex[i] = loadIcon("icons/", btnNames[i]);
        }
    }

    return true;
}

void glfw_onMouseMoveOrbit(GLFWwindow* window, double posX, double posY) {
    static glm::vec2 lastMousePos = glm::vec2(0, 0);

    if (ImGui::GetIO().WantCaptureMouse) {
        lastMousePos.x = (float)posX;
        lastMousePos.y = (float)posY;
        return;
    }

    float deltaX = posX - lastMousePos.x;
    float deltaY = posY - lastMousePos.y;

    if(currentMainMode == REGISTRATION_MODE){

        FullSphereCamera* activeCamera = getActiveCamera(window);

        if (!isDragging) {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == 1 && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != 1) {
                activeCamera->Rotate(deltaX, deltaY);
            }
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
                float dx = posX - lastMousePos.x;
                float dy = lastMousePos.y - posY;
                activeCamera->Pan(dx, dy);
            }
        }
        if (isDragging && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            float dx = ((float)posX - lastMousePos.x) * activeCamera->LIGHT_MOUSE_SENSITIVITY;
            float dy = (lastMousePos.y - (float)posY) * activeCamera->LIGHT_MOUSE_SENSITIVITY;
            glm::vec3 moveDirection = activeCamera->cameraRight * dx + activeCamera->cameraUp * dy;
            translateAllMeshes(moveDirection);
        }
        if (isDragging && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == 1 && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != 1) {
            float rotX = ((float)posY - lastMousePos.y) * 0.01f;
            float rotY = ((float)posX - lastMousePos.x) * 0.01f;
            glm::vec3 center = liverMesh3D->calcCenter();
            rotateAllMeshes(center, activeCamera->cameraRight, rotX, activeCamera->cameraUp, rotY);
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == 1 && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == 1) {
            glm::vec3 movement = activeCamera->cameraDirection * ((float)posY - lastMousePos.y) * activeCamera->LIGHT_MOUSE_SENSITIVITY;
            translateAllMeshes(movement);
        }
    }

    if(currentMainMode == DEFORM_MODE){

        bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

        if (isDragging && deformHandlPlace.state == DeformHandlPlaceData::RIGID_MODE && multiBody != nullptr) {
            if (leftDown && rightDown) {
                glm::vec3 movement = OrbitCam.cameraDirection * (-deltaY) * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
                multiBody->rigidTranslate(movement);
            } else if (leftDown && !rightDown) {
                float rotX = deltaY * 0.01f;
                float rotY = deltaX * 0.01f;
                if (std::abs(rotX) > 1e-5f)
                    multiBody->rigidRotateAroundCenter(OrbitCam.cameraRight, rotX);
                if (std::abs(rotY) > 1e-5f)
                    multiBody->rigidRotateAroundCenter(OrbitCam.cameraUp, rotY);
            } else if (rightDown && !leftDown) {
                float dx = deltaX * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
                float dy = -deltaY * OrbitCam.LIGHT_MOUSE_SENSITIVITY;
                glm::vec3 move = OrbitCam.cameraRight * dx + OrbitCam.cameraUp * dy;
                multiBody->rigidTranslate(move);
            }
        }
        else if (isDragging && deformHandlPlace.state == DeformHandlPlaceData::DEFORM_MODE) {
            if (leftDown && !rightDown) {
                if (gGrabber != nullptr) {
                    gGrabber->moveGrab(posX, posY, 1.0f / 60.0f);
                }
            }
        }

        if (!isDragging) {
            if (leftDown && !rightDown) {
                OrbitCam.Rotate(deltaX, deltaY);
            }
            if (rightDown && !leftDown) {
                float dx = posX - lastMousePos.x;
                float dy = lastMousePos.y - posY;
                OrbitCam.Pan(dx, dy);
            }
        }
    }

    lastMousePos.x = (float)posX;
    lastMousePos.y = (float)posY;
}

/* ===========================================================================
 * Scroll wheel handler — Mode 2 layout
 * ===========================================================================
 *   Right-button + scroll : scale meshes (unchanged behavior)
 *   Shift + scroll        : change camera DISTANCE (gRadius)
 *                            — moves the camera physically along its view ray.
 *                              Changes occlusion / parallax / perspective
 *                              foreshortening. Used historically by all
 *                              workflows that pre-date Mode 2.
 *   Plain scroll          : change camera FOV (focal length fy/fx)
 *                            — keeps camera position fixed; only scales
 *                              the angular size of the scene on the image
 *                              plane. The 3D-to-2D projection geometry
 *                              between liver mesh and the image plane is
 *                              invariant under this operation, so the
 *                              relative silhouette geometry (and hence IoU
 *                              for any registered pose) does NOT change.
 *                              Preferred for IoU-driven workflows where the
 *                              user just wants to "frame" the working area.
 *
 * Direction conventions (matched to historical scroll-up = zoom-in):
 *   ZOOM_SENSITIVITY = -1.0, so plain old code did:
 *     scroll up (deltaY > 0) -> gRadius += deltaY * (-1)  -> gRadius DOWN
 *                            -> camera moves CLOSER -> visual zoom IN
 *   FOV variant must match: scroll up -> FOV NARROWER -> visual zoom IN
 *     - In intrinsic mode (useIntrinsics == true, which is the active mode
 *       since main() calls OrbitCam.setIntrinsics(800, 800, ...)),
 *       narrower FOV means LARGER fy (and proportionally fx to preserve
 *       aspect ratio). We multiply both by 1.1^deltaY.
 *     - In legacy gFOV mode, narrower means SMALLER gFOV in degrees,
 *       so subtract.
 *
 * Clamping rationale:
 *   For window height 720 px, fy in [300, 3000] yields vertical FOV in
 *   roughly [14 deg, 100 deg] — covers everything from telephoto to
 *   wide-angle without wraparound or numerical degeneracy.
 *   gRadius clamp [2, 80] preserved verbatim from previous behavior.
 * =========================================================================== */
void glfw_onMouseScroll(GLFWwindow* window, double deltaX, double deltaY) {
    if (ImGui::GetIO().WantCaptureMouse) return;

    FullSphereCamera* activeCamera = getActiveCamera(window);

    const bool rightHeld =
        (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    const bool shiftHeld =
        (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS) ||
        (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    if (rightHeld) {
        /* Right-button + scroll : mesh scale (unchanged) */
        glm::vec3 center = liverMesh3D->calcCenter();
        float scale = calcScaleFactor(deltaY, scaleSpeed);

        if (currentMainMode == REGISTRATION_MODE) {
            scaleAllMeshes(center, scale);
        }
        if (currentMainMode == DEFORM_MODE) {
            liverMesh3D->scaleAround(center, scale);
            setUp(*liverMesh3D);
        }
        if (!registrationHandle.objectPoints.empty()) {
            scaleRegistrationPoints(center, scale);
        }
        return;
    }

    if (shiftHeld) {
        /* Shift + scroll : camera DISTANCE (gRadius). Old default behavior. */
        activeCamera->gRadius += deltaY * activeCamera->ZOOM_SENSITIVITY;
        activeCamera->gRadius = glm::clamp(activeCamera->gRadius, 2.0f, 80.0f);
        std::cout << "[Scroll] distance (Shift+scroll): radius="
                  << activeCamera->gRadius << std::endl;
        return;
    }

    /* Plain scroll : FOV change (focal length). Geometry-preserving zoom. */
    if (activeCamera->useIntrinsics) {
        /* Intrinsic mode: change fy / fx multiplicatively, keep aspect ratio. */
        const float factor = std::pow(1.1f, (float)deltaY);
        float newFy = activeCamera->fy * factor;
        float newFx = activeCamera->fx * factor;
        newFy = glm::clamp(newFy, 300.0f, 3000.0f);
        newFx = glm::clamp(newFx, 300.0f, 3000.0f);
        activeCamera->fy = newFy;
        activeCamera->fx = newFx;
        const float fovY = activeCamera->calculateFOVFromIntrinsics();
        activeCamera->gFOV = fovY;   /* keep gFOV in sync for any consumer */
        std::cout << "[Scroll] FOV (geometry-preserving): "
                  << std::fixed << std::setprecision(1) << fovY
                  << " deg  (fy=" << newFy << ")"
                  << std::defaultfloat << std::setprecision(6) << std::endl;
    } else {
        /* Legacy FOV mode: directly modify gFOV in degrees. */
        activeCamera->gFOV -= (float)deltaY * 2.0f;   /* 2 deg per notch */
        activeCamera->gFOV = glm::clamp(activeCamera->gFOV, 14.0f, 100.0f);
        std::cout << "[Scroll] FOV (gFOV mode): "
                  << activeCamera->gFOV << " deg" << std::endl;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    if(currentMainMode == DEFORM_MODE) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                if (deformHandlPlace.state == DeformHandlPlaceData::HANDLE_PLACE_MODE) {

                    if (multiBody->handleGroups.size() >= SoftBody::MAX_HANDLE_GROUPS) {
                        std::cout << "Maximum " << SoftBody::MAX_HANDLE_GROUPS
                                  << " handle groups reached. Press C to clear." << std::endl;
                        return;
                    }

                    int expectedIndex = deformHandlPlace.softbodyPoints.size();
                    std::cout << ">>> Selecting handle point #" << (expectedIndex + 1)
                              << "/" << SoftBody::MAX_HANDLE_GROUPS << std::endl;

                    gGrabber->placeSphere(xpos, ypos, gGroupRadius);

                    if (hit_index >= 0) {
                        deformHandlPlace.softbodyPoints.push_back(hit_position);

                        glm::vec3 color = getPointColor(expectedIndex, false);
                        std::cout << "Handle group " << deformHandlPlace.softbodyPoints.size()
                                  << " [Color: R=" << color.r << " G=" << color.g << " B=" << color.b << "]"
                                  << " created" << std::endl;

                        if (deformHandlPlace.softbodyPoints.size() >= SoftBody::MAX_HANDLE_GROUPS) {
                            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
                            multiBody->setRigidMode(false);
                            std::cout << "\n=== Maximum groups reached. Switched to DEFORM MODE ===" << std::endl;
                        }
                    }

                    isDragging = false;
                    hit_index = -1;
                }
                else if (deformHandlPlace.state == DeformHandlPlaceData::RIGID_MODE) {
                    if (gGrabber && gGrabber->hitTest(xpos, ypos))
                        isDragging = true;
                }
                else if (deformHandlPlace.state == DeformHandlPlaceData::DEFORM_MODE) {
                    gGrabber->startGrab(xpos, ypos);
                } else if(deformHandlPlace.state == DeformHandlPlaceData::PLANECUT_MODE){
                    std::cout << "Pefortm Cutter FindHit" << std::endl;
                    if (cutterMesh) {
                        FindHit(xpos, ypos, cutterMesh->mVertices, cutterMesh->mIndices);
                    }
                }
            }
            else if (action == GLFW_RELEASE) {
                if (deformHandlPlace.state == DeformHandlPlaceData::HANDLE_PLACE_MODE) {
                    isDragging = false;
                    hit_index = -1;
                }
                else if (deformHandlPlace.state == DeformHandlPlaceData::RIGID_MODE) {
                    isDragging = false;
                }
                else {
                    hit_index = -1;
                    isDragging = false;
                    gGrabber->endGrab();
                }
            }
        }

        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                if(deformHandlPlace.state == DeformHandlPlaceData::RIGID_MODE){
                    if (gGrabber && gGrabber->hitTest(xpos, ypos))
                        isDragging = true;
                } else if(deformHandlPlace.state == DeformHandlPlaceData::PLANECUT_MODE){
                    std::cout << "triMesh Find Hit" << std::endl;
                    if (cutterMesh) {
                        FindHit(xpos, ypos, cutterMesh->mVertices, cutterMesh->mIndices);
                    }
                }
            } else if (action == GLFW_RELEASE) {
                if (deformHandlPlace.state == DeformHandlPlaceData::RIGID_MODE) {
                    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
                        isDragging = false;
                } else {
                    hit_index = -1;
                    isDragging = false;
                }
            }
        }
    }

    if(currentMainMode == REGISTRATION_MODE) {

        bool isLeftScreen = false;
        bool isRightScreen = false;
        FullSphereCamera* activeCamera = getActiveCameraWithSide(window, isLeftScreen, isRightScreen);

        if (gCameraPreview.frozen && action == GLFW_PRESS
            && (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT)) {

            if (depthSplitScreenMode && !isRightScreen) {
                std::cout << "[SegPoint] Click on RIGHT screen to add points" << std::endl;
                return;
            }

            glm::vec3 hitPos;
            if (depthSplitScreenMode) {
                int halfW = gWindowWidth / 2;
                float localX = (float)xpos - halfW;
                hitPos = pickPointOnBoardWithCamera(localX, ypos, &OrbitCamRight_Screen, halfW, gWindowHeight);
            } else {
                hitPos = pickPointOnBoardWithCamera(xpos, ypos, activeCamera, gWindowWidth, gWindowHeight);
            }

            if (hitPos != glm::vec3(-999)) {
                int pixelX, pixelY;
                if (convert3DToImagePixel(hitPos, screenMesh, pixelX, pixelY)) {
                    if (button == GLFW_MOUSE_BUTTON_LEFT) {
                        gUserSegPoints.emplace_back(
                            static_cast<float>(pixelX),
                            static_cast<float>(pixelY), true);
                        gUserSegPoints3D.push_back(hitPos);
                        gUserSegPointsFG.push_back(true);
                        std::cout << "[SegPoint] FG(object) #" << gUserSegPoints.size()
                                  << " at 2D(" << pixelX << ", " << pixelY << ")" << std::endl;
                    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                        gUserSegPoints.emplace_back(
                            static_cast<float>(pixelX),
                            static_cast<float>(pixelY), false);
                        gUserSegPoints3D.push_back(hitPos);
                        gUserSegPointsFG.push_back(false);
                        std::cout << "[SegPoint] BG(background) #" << gUserSegPoints.size()
                                  << " at 2D(" << pixelX << ", " << pixelY << ")" << std::endl;
                    }

                    std::cout << "[SegPoint] Total: " << gUserSegPoints.size() << " points (";
                    int fgCount = 0, bgCount = 0;
                    for (const auto& p : gUserSegPoints) {
                        if (p.isForeground) fgCount++; else bgCount++;
                    }
                    std::cout << fgCount << " FG, " << bgCount << " BG)" << std::endl;
                }
            } else {
                std::cout << "[SegPoint] No hit on screenMesh" << std::endl;
            }
            return;
        }

        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {

            if (registrationHandle.state == RegistrationData::SELECTING_BOARD_POINTS) {

                if (splitScreenMode && !isRightScreen) {
                    std::cout << "Board point selection is only available on the right screen (texture view)" << std::endl;
                    return;
                }

                std::cout << ">>> Selecting board point #" << (registrationHandle.boardPoints.size() + 1)
                          << " of " << registrationHandle.targetPointCount << "..." << std::endl;

                glm::vec3 boardPoint;
                if (splitScreenMode) {
                    double adjustedX = xpos - gWindowWidth / 2.0;
                    boardPoint = pickPointOnBoardWithCamera(adjustedX, ypos, activeCamera,
                                                            gWindowWidth/2, gWindowHeight);
                } else {
                    boardPoint = pickPointOnBoardWithCamera(xpos, ypos, activeCamera,
                                                            gWindowWidth, gWindowHeight);
                }

                if (boardPoint != glm::vec3(-999)) {
                    registrationHandle.boardPoints.push_back(boardPoint);

                    int pointIndex = registrationHandle.boardPoints.size() - 1;
                    glm::vec3 color = getPointColor(pointIndex, true);
                    std::cout << "Board point " << registrationHandle.boardPoints.size()
                              << " [Color: R=" << color.r << " G=" << color.g << " B=" << color.b << "]"
                              << ": (" << boardPoint.x << ", " << boardPoint.y
                              << ", " << boardPoint.z << ")" << std::endl;

                    if (registrationHandle.boardPoints.size() >= registrationHandle.targetPointCount) {
                        registrationHandle.state = RegistrationData::SELECTING_OBJECT_POINTS;
                        std::cout << "\n=== SWITCHED TO OBJECT SELECTION MODE ===" << std::endl;
                        std::cout << "Select " << registrationHandle.targetPointCount
                                  << " points on the 3D object in THE SAME ORDER!" << std::endl;
                        if (splitScreenMode) {
                            std::cout << "NOTE: Switch to the LEFT screen for object selection" << std::endl;
                        }
                    } else {
                        std::cout << "Select " << (registrationHandle.targetPointCount - registrationHandle.boardPoints.size())
                        << " more board points" << std::endl;
                    }
                }
            }

            else if (registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS) {

                if (splitScreenMode && !isLeftScreen) {
                    std::cout << "Object point selection is only available on the left screen (liver view)" << std::endl;
                    return;
                }

                int expectedIndex = registrationHandle.objectPoints.size();
                std::cout << ">>> Selecting object point #" << (expectedIndex + 1)
                          << " (corresponds to board point #" << (expectedIndex + 1) << ")..." << std::endl;

                if (splitScreenMode) {
                    FindHitWithCamera(xpos, ypos, liverMesh3D->mVertices, liverMesh3D->mIndices,
                                      activeCamera, gWindowWidth/2, gWindowHeight);
                } else {
                    FindHitWithCamera(xpos, ypos, liverMesh3D->mVertices, liverMesh3D->mIndices,
                                      activeCamera, gWindowWidth, gWindowHeight);
                }

                if (hit_index >= 0) {
                    registrationHandle.objectPoints.push_back(hit_position);

                    glm::vec3 color = getPointColor(expectedIndex, false);
                    std::cout << "Object point " << registrationHandle.objectPoints.size()
                              << " [Color: R=" << color.r << " G=" << color.g << " B=" << color.b << "]"
                              << ": (" << hit_position.x << ", " << hit_position.y
                              << ", " << hit_position.z << ")" << std::endl;

                    if (registrationHandle.objectPoints.size() >= registrationHandle.boardPoints.size()) {
                        registrationHandle.state = RegistrationData::READY_TO_REGISTER;
                        std::cout << "\n=== READY TO REGISTER ===" << std::endl;
                        std::cout << "Point correspondences:" << std::endl;
                        for (size_t i = 0; i < registrationHandle.boardPoints.size(); i++) {
                            std::cout << "  Pair " << (i+1) << ": Board->Object" << std::endl;
                        }
                        std::cout << "Press H to execute registration" << std::endl;
                    }
                }

                isDragging = false;
                hit_index = -1;
            }

            else if (registrationHandle.state == RegistrationData::IDLE ||
                     registrationHandle.state == RegistrationData::REGISTERED) {

                std::cout << "Normal mode: Find Hit" << std::endl;

                if (splitScreenMode && isRightScreen) {
                    std::cout << "Liver mesh manipulation is only available on the left screen" << std::endl;
                    return;
                }

                if (splitScreenMode) {
                    FindHitWithCamera(xpos, ypos, liverMesh3D->mVertices, liverMesh3D->mIndices,
                                      activeCamera, gWindowWidth/2, gWindowHeight);
                } else {
                    FindHitWithCamera(xpos, ypos, liverMesh3D->mVertices, liverMesh3D->mIndices,
                                      activeCamera, gWindowWidth, gWindowHeight);
                }
            }
        }

        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (registrationHandle.state != RegistrationData::IDLE &&
                registrationHandle.state != RegistrationData::REGISTERED) {
                std::cout << "Right click disabled during registration" << std::endl;
                return;
            }

            if (splitScreenMode && isRightScreen) {
                std::cout << "Right click manipulation is only available on the left screen" << std::endl;
                return;
            }

            std::vector<mCutMesh*> meshesToHit = {
                liverMesh3D,
                portalMesh3D,
                veinMesh3D,
                tumorMesh3D,
                segmentMesh3D,
                gbMesh3D
            };

            if (splitScreenMode) {
                FindHitWithCameraMultipleMeshes(xpos, ypos, meshesToHit,
                                                activeCamera, gWindowWidth/2, gWindowHeight);
            } else {
                FindHitWithCameraMultipleMeshes(xpos, ypos, meshesToHit,
                                                activeCamera, gWindowWidth, gWindowHeight);
            }

        }
        else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
            hit_index = -1;
            isDragging = false;
        }

        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (registrationHandle.state != RegistrationData::IDLE &&
                registrationHandle.state != RegistrationData::REGISTERED) {
                registrationHandle.reset();
                std::cout << "=== Registration cancelled ===" << std::endl;
            }
        }

    }

}

void glfw_onKey(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;

    case GLFW_KEY_0:
        if (multiBody) {
            multiBody->toggleTetMeshVisible();
            std::cout << "[Wireframe] TetMesh visible: "
                      << (multiBody->isTetMeshVisible() ? "ON" : "OFF") << std::endl;
        }
        break;

    case GLFW_KEY_1:
        if (currentMainMode == DEFORM_MODE) {
            glm::vec3 initCamPos(0.0f, 0.0f, OrbitCam.InitialRadius);
            glm::vec3 initCamTgt(0.0f, 0.0f, 0.0f);
            std::cout << "[AutoDeform Step1] using INITIAL camera pose ("
                      << initCamPos.x << ", " << initCamPos.y << ", " << initCamPos.z
                      << ") -> (0,0,0)" << std::endl;
            AutoDeform::classifySrcVisibility(
                gAutoDeform, liverMesh3D, screenMesh,
                gGridWidth, gGridHeight(),
                initCamPos, initCamTgt, gDepthScale);
        }
        break;

    case GLFW_KEY_2:
        if (currentMainMode == DEFORM_MODE) {
            AutoDeform::extractCorrespondences(
                gAutoDeform, liverMesh3D, screenMesh,
                gGridWidth, gGridHeight(), gDepthScale, 1.0f);
        }
        break;

    case GLFW_KEY_3:
        if (currentMainMode == DEFORM_MODE) {
            if (gAutoDeform.correspondences.empty()) {
                AutoDeform::extractCorrespondences(
                    gAutoDeform, liverMesh3D, screenMesh,
                    gGridWidth, gGridHeight(), gDepthScale, 1.0f);
            }
            AutoDeform::classify(gAutoDeform, 1.0f, 3.0f);
        }
        break;

    case GLFW_KEY_8:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            multiBody->clearAttachmentConstraints();
            const auto& pos = multiBody->getPositions();
            size_t N = multiBody->getNumParticles();
            if (N == 0) {
                std::cout << "[Attachment Test] no particles" << std::endl;
                break;
            }
            int step = std::max<int>(1, static_cast<int>(N) / 20);
            int count = 0;
            for (size_t i = 0; i < N && count < 20; i += step, count++) {
                glm::vec3 p(pos[i*3], pos[i*3+1], pos[i*3+2]);
                glm::vec3 target = p + glm::vec3(0.0f, 0.3f, 0.0f);
                float compliance = (count < 5) ? 0.0f : 1e-6f;
                multiBody->addAttachmentConstraint(static_cast<int>(i), target, compliance);
            }
            std::cout << "[Attachment Test] created "
                      << multiBody->getNumAttachments()
                      << " attachments (first 5 rigid, rest soft, +Y 0.3 offset)"
                      << std::endl;
        }
        break;

    case GLFW_KEY_9:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            multiBody->clearAttachmentConstraints();
            std::cout << "[Attachment Test] cleared all attachments" << std::endl;
        }
        break;

    case GLFW_KEY_4:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            if (gAutoDeform.correspondences.empty()) {
                AutoDeform::extractCorrespondences(
                    gAutoDeform, liverMesh3D, screenMesh,
                    gGridWidth, gGridHeight(), gDepthScale, 1.0f);
            }
            const auto& visVerts  = multiBody->getVisPositions(0);
            const auto& visTriIds = multiBody->getVisSurfaceTriIds(0);
            AutoDeform::computeFieldOnVisMesh(gAutoDeform, visVerts, visTriIds, 8, 0.0f, true, 10, 0.5f);
        }
        break;

    case GLFW_KEY_5:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            if (!gAutoDeform.fieldReady) {
                if (gAutoDeform.correspondences.empty()) {
                    AutoDeform::extractCorrespondences(
                        gAutoDeform, liverMesh3D, screenMesh,
                        gGridWidth, gGridHeight(), gDepthScale, 1.0f);
                }
                const auto& visVerts  = multiBody->getVisPositions(0);
                const auto& visTriIds = multiBody->getVisSurfaceTriIds(0);
                AutoDeform::computeFieldOnVisMesh(gAutoDeform, visVerts, visTriIds, 8, 0.0f, true, 10, 0.5f);
            }
            const auto& p = AutoDeform::getPresets()[gAutoDeformPresetIdx];
            std::cout << "[AutoDeform] Using preset: " << p.name << std::endl;
            AutoDeform::generateHandles(
                gAutoDeform,
                p.K_fix, p.K_move,
                p.tauLowScale, p.tauHighScale,
                p.rFixScale, p.rMoveScale,
                p.minSepScale);

            std::cout << "--- [Debug Key5] after generateHandles ---" << std::endl;
            for (size_t i = 0; i < gAutoDeform.fixHandles.size(); i++) {
                const auto& h = gAutoDeform.fixHandles[i];
                std::cout << "  fix[" << i << "]"
                          << " center=(" << h.center.x << "," << h.center.y << "," << h.center.z << ")"
                          << " radius=" << h.radius
                          << std::endl;
            }
            for (size_t i = 0; i < gAutoDeform.moveHandles.size(); i++) {
                const auto& h = gAutoDeform.moveHandles[i];
                glm::vec3 disp = h.target - h.center;
                std::cout << "  move[" << i << "]"
                          << " center=(" << h.center.x << "," << h.center.y << "," << h.center.z << ")"
                          << " target=(" << h.target.x << "," << h.target.y << "," << h.target.z << ")"
                          << " |disp|=" << glm::length(disp)
                          << " radius=" << h.radius
                          << std::endl;
            }

            gAutoCtrl.initialize(multiBody, gAutoDeform);
            gAutoDeformFirstCommit = false;
            multiBody->setRigidMode(false);
            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
            std::cout << "[Key5] auto-switched to DEFORM_MODE (rigidMode=false)" << std::endl;
        }
        break;

    case GLFW_KEY_6:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            if (gAutoDeform.stage < 5 || gAutoCtrl.numMove() == 0) {
                std::cout << "[AutoDeform] Run handle gen (key 5) first" << std::endl;
                break;
            }
            if (gAutoCtrl.activeMoveIdx() < 0) {
                std::cout << "[Key6] no active move (press A to select first)" << std::endl;
                break;
            }
            std::cout << "[Key6 debug] rigid=" << multiBody->isRigidMode()
                      << " att=" << multiBody->getNumAttachments()
                      << " state=" << (int)deformHandlPlace.state << std::endl;

            bool isFirst = !gAutoDeformFirstCommit;
            if (isFirst) {
                gSnapBeforePositions    = multiBody->getPositions();
                gSnapBeforeVisPositions = multiBody->getAllVisPositions();
                gSnapBeforeValid = true;
                gSnapAfterValid  = false;
                gShowingAfter    = true;
                const auto& visVerts0 = multiBody->getVisPositions(0);
                AutoDeform::measureRMSE(
                    gAutoDeform, visVerts0, screenMesh,
                    gGridWidth, gGridHeight(), gDepthScale,
                    /*storeAsBefore=*/true, 1.0f, /*verbose=*/false);
                std::cout << "[Key6] baseline RMSE=" << gAutoDeform.rmseBeforeDeform << std::endl;
                gAutoDeformFirstCommit = true;
            }

            int activeIdx = gAutoCtrl.activeMoveIdx();
            float prevProg = gAutoCtrl.moveHandle(activeIdx).progress;
            float prevRMSE = gAutoDeform.rmseLastMeasured;
            bool changed = gAutoCtrl.stepActive(multiBody, +gMoveScale);
            if (!changed) {
                std::cout << "[Key6] slot=" << activeIdx << " already at goal, no-op" << std::endl;
                break;
            }
            gAutoCtrl.runBoost(multiBody, gAutoDeformBoostIter, gAutoDeformBoostDamping);

            float newProg = gAutoCtrl.moveHandle(activeIdx).progress;
            const auto& visVertsForRMSE = multiBody->getVisPositions(0);
            float immRMSE = AutoDeform::measureRMSE(
                gAutoDeform, visVertsForRMSE, screenMesh,
                gGridWidth, gGridHeight(), gDepthScale,
                /*storeAsBefore=*/false, 1.0f, /*verbose=*/false);
            float stepD = (prevRMSE > 0.0f) ? (immRMSE - prevRMSE) : 0.0f;
            float baseD = (gAutoDeform.rmseBeforeDeform > 0.0f) ? (immRMSE - gAutoDeform.rmseBeforeDeform) : 0.0f;
            std::cout << "[6] move=" << activeIdx << " progress " << prevProg << " -> " << newProg
                      << "  RMSE=" << immRMSE
                      << "  step:" << (stepD>=0?"+":"") << stepD << (stepD<0?" DOWN":" UP")
                      << "  base:" << (baseD>=0?"+":"") << baseD << (baseD<0?" DOWN":" UP")
                      << std::endl;

            gPendingRmseImmediate = immRMSE;
            gPendingRmseFrames    = 60;
            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
        }
        break;

    case GLFW_KEY_7:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            if (!gSnapBeforeValid) {
                const auto& visVerts0 = multiBody->getVisPositions(0);
                AutoDeform::measureRMSE(
                    gAutoDeform, visVerts0, screenMesh,
                    gGridWidth, gGridHeight(), gDepthScale,
                    /*storeAsBefore=*/false, 1.0f, true);
                std::cout << "[AutoDeform] No snapshot available (press key 6 first for toggle)" << std::endl;
                break;
            }

            bool firstCapture = false;
            if (!gSnapAfterValid) {
                gSnapAfterPositions    = multiBody->getPositions();
                gSnapAfterVisPositions = multiBody->getAllVisPositions();
                gSnapAfterValid = true;
                firstCapture = true;
                float maxD = 0.0f, sumD = 0.0f; int cnt = 0;
                if (gSnapBeforeValid && gSnapAfterPositions.size() == gSnapBeforePositions.size()) {
                    for (size_t k = 0; k + 2 < gSnapAfterPositions.size(); k += 3) {
                        glm::vec3 a(gSnapAfterPositions[k], gSnapAfterPositions[k+1], gSnapAfterPositions[k+2]);
                        glm::vec3 b(gSnapBeforePositions[k], gSnapBeforePositions[k+1], gSnapBeforePositions[k+2]);
                        float d = glm::length(a - b);
                        if (d > maxD) maxD = d;
                        sumD += d; cnt++;
                    }
                }
                std::cout << "[AutoDeform] AFTER snapshot captured" << std::endl;
                std::cout << "[DBG key7 capture] rigid=" << multiBody->isRigidMode()
                          << " att=" << multiBody->getNumAttachments()
                          << " maxParticleDisp=" << maxD
                          << " avgParticleDisp=" << (cnt > 0 ? sumD / cnt : 0.0f)
                          << " sub=" << (int)deformHandlPlace.state << std::endl;
            }

            multiBody->setRigidMode(true);
            if (!firstCapture) gShowingAfter = !gShowingAfter;
            else               gShowingAfter = true;
            if (gShowingAfter) {
                multiBody->setPositions(gSnapAfterPositions);
                multiBody->setAllVisPositions(gSnapAfterVisPositions);
                std::cout << "[Toggle] Showing AFTER (deformed)" << (firstCapture ? " [first capture, hold]" : "") << std::endl;
            } else {
                multiBody->setPositions(gSnapBeforePositions);
                multiBody->setAllVisPositions(gSnapBeforeVisPositions);
                std::cout << "[Toggle] Showing BEFORE (original)" << std::endl;
            }
            multiBody->updateTetMeshes();
            multiBody->updateVisMeshes();

            const auto& visVerts0 = multiBody->getVisPositions(0);
            AutoDeform::measureRMSE(
                gAutoDeform, visVerts0, screenMesh,
                gGridWidth, gGridHeight(), gDepthScale,
                /*storeAsBefore=*/false, 1.0f, true);
        }
        break;

    case GLFW_KEY_Z:
        if (currentMainMode == REGISTRATION_MODE && gCameraPreview.frozen) {
            if (!gUserSegPoints.empty()) {
                auto& removed = gUserSegPoints.back();
                std::cout << "[SegPoint] Undo: removed "
                          << (removed.isForeground ? "FG" : "BG")
                          << " at 2D(" << removed.x << ", " << removed.y << ")" << std::endl;
                gUserSegPoints.pop_back();
                gUserSegPoints3D.pop_back();
                gUserSegPointsFG.pop_back();
                std::cout << "[SegPoint] Remaining: " << gUserSegPoints.size() << " points" << std::endl;
            } else {
                std::cout << "[SegPoint] Nothing to undo" << std::endl;
            }
        }
        break;

    case GLFW_KEY_UP:
        if (currentMainMode == REGISTRATION_MODE) {
            g_voxelSize += 0.05f;
            std::cout << "[VoxelSize] " << g_voxelSize << std::endl;
        }
        break;

    case GLFW_KEY_DOWN:
        if (currentMainMode == REGISTRATION_MODE) {
            g_voxelSize = std::max(0.0f, g_voxelSize - 0.05f);
            std::cout << "[VoxelSize] " << g_voxelSize << std::endl;
        }
        break;

    case GLFW_KEY_COMMA:
        if (currentMainMode == REGISTRATION_MODE) {
            int prevGW = gGridWidth;
            gGridWidth = std::max(64, gGridWidth / 2);
            if (gGridWidth != prevGW) {
                regenerateDepthMeshAuto(screenMesh, gDepthScale, gMeshScale);
                int gh = gGridWidth * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
                std::cout << "[Grid] " << prevGW << " -> " << gGridWidth
                          << " (" << (gGridWidth+1)*(gh+1) << " vertices)" << std::endl;
            }
        }
        break;

    case GLFW_KEY_PERIOD:
        if (currentMainMode == REGISTRATION_MODE) {
            int prevGW = gGridWidth;
            gGridWidth = std::min(512, gGridWidth * 2);
            if (gGridWidth != prevGW) {
                regenerateDepthMeshAuto(screenMesh, gDepthScale, gMeshScale);
                int gh = gGridWidth * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
                std::cout << "[Grid] " << prevGW << " -> " << gGridWidth
                          << " (" << (gGridWidth+1)*(gh+1) << " vertices)" << std::endl;
            }
        }
        break;

    case GLFW_KEY_U:
        if (currentMainMode == REGISTRATION_MODE) {
            if (gCameraPreview.active) {
                clearSegPoints();
                gCameraPreview.captureAndFreeze(screenMesh);
                depthSplitScreenMode = true;
                splitScreenMode = true;
                OrbitCamLeft_Target = OrbitCam;
                OrbitCamRight_Screen = OrbitCam;
                OrbitCamLeft_Target.currentTarget = TARGET_LIVER;
                OrbitCamLeft_Target.cx = (gWindowWidth / 2) / 2.0f;
                OrbitCamLeft_Target.cy = gWindowHeight / 2.0f;
                OrbitCamRight_Screen.currentTarget = TARGET_TEXTURE;
                OrbitCamRight_Screen.gRadius = OrbitCam.InitialRadius * 2.0f;
                OrbitCamRight_Screen.cx = (gWindowWidth / 2) / 2.0f;
                OrbitCamRight_Screen.cy = gWindowHeight / 2.0f;
                std::cout << "[SegPoint] Left-click = FG(object), Right-click = BG(background)" << std::endl;
                std::cout << "[SegPoint] Z key = Undo last point" << std::endl;
                std::cout << "[SegPoint] Press I to run depth with segmentation" << std::endl;
                std::cout << "[SegPoint] Press K to run depth WITHOUT segmentation" << std::endl;
            } else if (gCameraPreview.frozen) {
                depthSplitScreenMode = false;
                splitScreenMode = false;
                gCameraPreview.clearFrozen();
                clearSegPoints();
                std::cout << "[SegPoint] Restarting camera..." << std::endl;
                gCameraPreview.start(screenMesh, 0, 1280, 720);
            } else {
                clearSegPoints();
                std::cout << "[SegPoint] Starting camera. Press U again to freeze." << std::endl;
                std::cout << "[SegPoint] After freeze: I=with seg, K=without seg" << std::endl;
                gCameraPreview.start(screenMesh, 0, 1280, 720);
            }
        }
        break;

    case GLFW_KEY_I:
        if (currentMainMode == REGISTRATION_MODE) {
            showProgressOverlay(0.05f, "Preparing depth...");
            std::vector<DepthRunnerPoint> segPoints;
            if (gUserSegPoints.empty()) {
                segPoints = createDefaultSegPoints(
                    screenMesh->loadedImageWidth,
                    screenMesh->loadedImageHeight);
                std::cout << "[Seg] Using DEFAULT points (center=FG + 4corners=BG)" << std::endl;
            } else {
                segPoints = gUserSegPoints;
                std::cout << "[Seg] Using " << segPoints.size() << " USER-SELECTED points" << std::endl;
            }

            if (gCameraPreview.frozen) {
                gCameraPreview.runDepthFromFrozen(gDepthRunner, screenMesh, segPoints, showProgressOverlay);
            } else if (gCameraPreview.active) {
                gCameraPreview.captureAndRunDepthWithPoints(gDepthRunner, screenMesh, segPoints, showProgressOverlay);
            } else if (gDepthRunner.isAvailable()) {
                DepthRunnerIntegration::updateScreenMeshDepth(
                    gDepthRunner, gDepthInputImage, screenMesh,
                    128, 10.0f, 0.3f, segPoints,
                    [](mCutMesh& mesh) { setUp(mesh); },
                    showProgressOverlay
                    );
            }

            showProgressOverlay(1.0f, "Depth complete!");
            resetBoundaryMap();
            gDepthScale = 0.3f;
            clearSegPoints();
            depthSplitScreenMode = false;
            splitScreenMode = false;
            registrationHandle.reset();
            registrationHandle.state = RegistrationData::IDLE;
            g_refineVertexIndices.clear();
            g_cluster1Points.clear();
            g_cluster2Points.clear();
            g_targetPoints.clear();
            g_showClusterVisualization = false;
            g_showCorrespondencePoints = false;
        }
        break;

    case GLFW_KEY_K:
        if (currentMainMode == REGISTRATION_MODE) {
            std::cout << "[Seg] Key K: Depth-only mode (NO segmentation)" << std::endl;
            showProgressOverlay(0.05f, "Depth-only mode...");

            if (gCameraPreview.frozen) {
                gCameraPreview.runDepthFullFromFrozen(gDepthRunner, screenMesh, showProgressOverlay);
            } else if (gCameraPreview.active) {
                gCameraPreview.captureAndFreeze(screenMesh);
                gCameraPreview.runDepthFullFromFrozen(gDepthRunner, screenMesh, showProgressOverlay);
            } else if (gDepthRunner.isAvailable()) {
                auto dummyPts = createDefaultSegPoints(
                    screenMesh->loadedImageWidth,
                    screenMesh->loadedImageHeight);
                DepthRunnerIntegration::updateScreenMeshDepthFullOnly(
                    gDepthRunner, gDepthInputImage, screenMesh,
                    128, 10.0f, 0.3f, dummyPts,
                    [](mCutMesh& mesh) { setUp(mesh); },
                    showProgressOverlay
                    );
            }

            showProgressOverlay(1.0f, "Depth complete!");
            resetBoundaryMap();
            gDepthScale = 0.3f;
            clearSegPoints();
            depthSplitScreenMode = false;
            splitScreenMode = false;
            registrationHandle.reset();
            registrationHandle.state = RegistrationData::IDLE;
            g_refineVertexIndices.clear();
            g_cluster1Points.clear();
            g_cluster2Points.clear();
            g_targetPoints.clear();
            g_showClusterVisualization = false;
            g_showCorrespondencePoints = false;
        }
        break;

    case GLFW_KEY_Y:
        if(currentMainMode == REGISTRATION_MODE) {

            splitScreenMode = false;

            registrationHandle.reset();
            registrationHandle.state = RegistrationData::IDLE;

            OrbitCam.cx = gWindowWidth / 2.0f;
            OrbitCam.cy = gWindowHeight / 2.0f;

            std::cout << "=== Custom Registration Started ===" << std::endl;

            std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D, tumorMesh3D, segmentMesh3D, gbMesh3D};
            std::vector<std::string> names = {"Liver", "Portal", "Vein", "Tumor", "Segment", "Gallbladder"};

            Reg3DCustom::performRegistrationMultiMeshWithScale(
                organs, names, screenMesh, OrbitCam.cameraPos,
                gGridWidth, gGridHeight(),
                15,
                0.005f,
                0.35f,
                true,
                0.03f,
                gDepthScale
                );

            std::cout << "=== Custom Registration Complete ===" << std::endl;
        }
        break;

    case GLFW_KEY_O:
        if (currentMainMode == REGISTRATION_MODE && (mode & GLFW_MOD_SHIFT)) {
            /* Shift+O: depth scale grid search + HemiAuto */
            std::cout << "\n=== Depth Scale Grid Search + HemiAuto (Shift+O) ===" << std::endl;
            poseAutoSaveBeforeRegistration();

            auto organs = getOrganList();
            if (g_initOrganVertices.empty() || g_initOrganVertices.size() != organs.size()) {
                std::cerr << "[Shift+O] No initial pose available." << std::endl;
                break;
            }

            const float scales[] = { 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f, 0.50f };
            const int   nScales  = (int)(sizeof(scales) / sizeof(scales[0]));

            float bestScale = gDepthScale;
            float bestRmse  = FLT_MAX;
            std::vector<std::vector<GLfloat>> bestVerts(organs.size());
            std::vector<std::vector<GLfloat>> bestNorms(organs.size());

            for (int si = 0; si < nScales; si++) {
                float ds = scales[si];

                for (size_t i = 0; i < organs.size(); i++) {
                    organs[i]->mVertices = g_initOrganVertices[i];
                    organs[i]->mNormals  = g_initOrganNormals[i];
                    setUp(*organs[i]);
                }
                regenerateDepthMeshAuto(screenMesh, ds, gMeshScale);
                resetRegistrationState();

                Reg3D::BVHTree bvh;
                bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
                auto vis = Reg3DCustom::extractVisibleVerticesCustom(
                    *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);
                if (vis.cloud->size() < 50) {
                    std::cout << "[Shift+O] scale=" << ds << "  skip (few points)" << std::endl;
                    continue;
                }
                g_cluster1Points      = vis.points;
                g_cluster2Points.clear();
                g_refineVertexIndices = vis.vertexIndices;
                computeIdealVoxelSizes();
                Reg3DCustom::performRegistrationSingleMesh(
                    organs, liverMesh3D, vis.vertexIndices,
                    screenMesh, OrbitCam.cameraPos,
                    gGridWidth, gGridHeight(), 15, 0.005f, 0.35f, true, 0.03f, ds, g_voxelSize);
                computeUnifiedMetrics();
                float rmse = registrationHandle.compRmse;
                std::cout << "[Shift+O] scale=" << ds << "  compRMSE=" << rmse << std::endl;

                if (rmse < bestRmse) {
                    bestRmse  = rmse;
                    bestScale = ds;
                    for (size_t i = 0; i < organs.size(); i++) {
                        bestVerts[i] = organs[i]->mVertices;
                        bestNorms[i] = organs[i]->mNormals;
                    }
                }
            }

            gDepthScale = bestScale;
            regenerateDepthMeshAuto(screenMesh, bestScale, gMeshScale);
            for (size_t i = 0; i < organs.size(); i++) {
                organs[i]->mVertices = bestVerts[i];
                organs[i]->mNormals  = bestNorms[i];
                setUp(*organs[i]);
            }
            computeUnifiedMetrics();
            std::cout << "[Shift+O] Best: scale=" << bestScale
                      << "  compRMSE=" << bestRmse << std::endl;

            g_bestSessionCompRmse = FLT_MAX;
            g_bestSessionVertices.clear();
            g_bestSessionNormals.clear();
            registrationHandle.state = RegistrationData::REGISTERED;
            registrationHandle.useRegistration = true;
            poseSaveToLibrary();

        } else if (currentMainMode == REGISTRATION_MODE) {
            /* Key O: camera view registration */
            std::cout << "\n============================================" << std::endl;
            std::cout << "  Camera View-Based Registration (Custom)" << std::endl;
            std::cout << "============================================\n" << std::endl;

            resetRegistrationState();

            Reg3D::BVHTree cameraBvhTree;
            cameraBvhTree.build(liverMesh3D->mVertices, liverMesh3D->mIndices);

            auto visibility = Reg3DCustom::extractVisibleVerticesCustom(
                *liverMesh3D, cameraBvhTree,
                OrbitCam.cameraPos, OrbitCam.cameraTarget);

            if (visibility.cloud->size() < 50) {
                std::cerr << "[X] ERROR: Not enough visible points ("
                          << visibility.cloud->size() << " < 50)" << std::endl;
                break;
            }

            g_cluster1Points = visibility.points;
            g_cluster2Points.clear();
            g_refineVertexIndices = visibility.vertexIndices;

            std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D, tumorMesh3D, segmentMesh3D, gbMesh3D};

            Reg3DCustom::performRegistrationSingleMesh(
                organs, liverMesh3D, visibility.vertexIndices,
                screenMesh, OrbitCam.cameraPos,
                gGridWidth, gGridHeight(),
                15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);

            std::cout << "=== Camera View Registration Complete ===" << std::endl;
        }
        break;

    case GLFW_KEY_N:
        if (currentMainMode == REGISTRATION_MODE) {
            if (registrationHandle.state == RegistrationData::REGISTERED) {
                if (g_refineVertexIndices.empty()) {
                    std::cerr << "[Refine] No visible vertex indices. Use Key O first." << std::endl;
                    break;
                }
                std::cout << "\n=== Normal-Compatible Refinement START ===" << std::endl;

                std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                                  tumorMesh3D, segmentMesh3D, gbMesh3D};
                NormalRefine::RefineParams params;

                if (NormalRefine::initRefine(g_refineState, liverMesh3D,
                                             g_refineVertexIndices,
                                             screenMesh, organs,
                                             gGridWidth, gGridHeight(), gDepthScale, params,
                                             NormalRefine::NORMAL_COMPAT)) {
                    computeUnifiedMetrics();
                    g_refineState.initialRMSE = registrationHandle.compRmse;
                    g_refineState.bestRMSE    = registrationHandle.compRmse;
                    std::cout << "[Refine] Unified initial RMSE: " << registrationHandle.compRmse << std::endl;
                    registrationHandle.state = RegistrationData::REFINING;
                } else {
                    std::cerr << "[Refine] Initialization failed" << std::endl;
                }

            } else if (registrationHandle.state == RegistrationData::REFINING) {
                g_refineState.active = false;
                registrationHandle.state = RegistrationData::REGISTERED;
                bool improved = g_refineState.bestRMSE < g_refineState.initialRMSE;
                g_refineState.restoreMeshes();
                if (improved) {
                    NormalRefine::applyIncrementalTransform(
                        g_refineState.bestCumulativeTransform,
                        g_refineState.organMeshes);
                }
                std::cout << "\n=== Normal-Compatible Refinement STOPPED ===" << std::endl;
                std::cout << "  Best RMSE: " << g_refineState.bestRMSE
                          << " (iter " << g_refineState.bestIteration << ")" << std::endl;
                std::cout << (improved ? "  >> Reverted to best" : "  >> Reverted to initial") << std::endl;
                registrationHandle.refineCount++;
                registrationHandle.refineInitialRMSE   = g_refineState.initialRMSE;
                registrationHandle.refineBestRMSE      = g_refineState.bestRMSE;
                registrationHandle.refineBestIteration  = g_refineState.bestIteration;
                computeUnifiedMetrics();
                poseSaveToLibrary();
            }
        }
        break;

    case GLFW_KEY_B:
        if (currentMainMode == REGISTRATION_MODE && (mode & GLFW_MOD_SHIFT)) {
            /* Shift+B: cycle through protocol B-trial initial poses
             * (B0 = TOP pure, B1..B4 = TOP + perturbation). Purely
             * visual/debug — does NOT run any registration. User can
             * then manually click Hemi Auto to reproduce a trial. */
            g_shiftBToggleIdx = (g_shiftBToggleIdx + 1) % 5;   /* 0..4 cycle */
            showTrialInitPose(g_shiftBToggleIdx);
            break;
        }
        if (currentMainMode == REGISTRATION_MODE) {
            if (registrationHandle.state == RegistrationData::REGISTERED) {
                if (g_refineVertexIndices.empty()) {
                    std::cerr << "[SRT-V] No visible vertex indices. Use Key O first." << std::endl;
                    break;
                }
                std::cout << "\n=== SRT Variance-Weighted Refinement START ===" << std::endl;

                std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                                  tumorMesh3D, segmentMesh3D, gbMesh3D};
                NormalRefine::RefineParams params;
                params.nSamples    = 11;
                params.sampleRange = 0.10f;
                params.srtSlope    = 0.02f;

                if (NormalRefine::initRefine(g_refineState, liverMesh3D,
                                             g_refineVertexIndices,
                                             screenMesh, organs,
                                             gGridWidth, gGridHeight(), gDepthScale, params,
                                             NormalRefine::SRT_VARIANCE)) {
                    computeUnifiedMetrics();
                    g_refineState.initialRMSE = registrationHandle.compRmse;
                    g_refineState.bestRMSE    = registrationHandle.compRmse;
                    std::cout << "[SRT-V] Unified initial RMSE: " << registrationHandle.compRmse << std::endl;
                    registrationHandle.state = RegistrationData::REFINING;
                } else {
                    std::cerr << "[SRT-V] Initialization failed" << std::endl;
                }

            } else if (registrationHandle.state == RegistrationData::REFINING) {
                g_refineState.active = false;
                registrationHandle.state = RegistrationData::REGISTERED;
                bool improved = g_refineState.bestRMSE < g_refineState.initialRMSE;
                g_refineState.restoreMeshes();
                if (improved) {
                    NormalRefine::applyIncrementalTransform(
                        g_refineState.bestCumulativeTransform,
                        g_refineState.organMeshes);
                }
                std::cout << "\n=== SRT Variance Refinement STOPPED ===" << std::endl;
                std::cout << "  Best RMSE: " << g_refineState.bestRMSE
                          << " (iter " << g_refineState.bestIteration << ")" << std::endl;
                std::cout << (improved ? "  >> Reverted to best" : "  >> Reverted to initial") << std::endl;
                registrationHandle.refineCount++;
                registrationHandle.refineInitialRMSE   = g_refineState.initialRMSE;
                registrationHandle.refineBestRMSE      = g_refineState.bestRMSE;
                registrationHandle.refineBestIteration  = g_refineState.bestIteration;
                computeUnifiedMetrics();
                poseSaveToLibrary();
            }
        }
        break;

    case GLFW_KEY_L:
        if (currentMainMode == REGISTRATION_MODE) {
            std::cout << "\n=== Raycast-Based Registration (Auto) ===\n" << std::endl;

            resetRegistrationState();

            Reg3D::BVHTree convergenceBvhTree;
            convergenceBvhTree.build(liverMesh3D->mVertices, liverMesh3D->mIndices);

            Reg3D::RaycastClusterer clusterer(convergenceBvhTree);
            auto clusteringResult = clusterer.performClustering(
                liverMesh3D->mVertices, liverMesh3D->mIndices);

            Reg3DCustom::NoOpen3DRegistration tempReg;
            auto targetCloud = tempReg.extractFrontFacePoints(*screenMesh, gGridWidth, gGridHeight(), gDepthScale);
            auto selectedClusters = Reg3DCustom::selectTop2ClustersCustom(clusteringResult, targetCloud);

            std::vector<size_t> mergedIndices;
            g_cluster1Points.clear();
            g_cluster2Points.clear();

            for (size_t c = 0; c < selectedClusters.size(); c++) {
                for (int idx : selectedClusters[c].visibleVertexIndices)
                    mergedIndices.push_back(static_cast<size_t>(idx));
                for (const auto& v : selectedClusters[c].visibleVertices) {
                    if (c == 0) g_cluster1Points.push_back(v);
                    else        g_cluster2Points.push_back(v);
                }
            }

            if (mergedIndices.size() < 50) {
                std::cerr << "[X] Not enough cluster vertices" << std::endl;
                break;
            }

            std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D, tumorMesh3D, segmentMesh3D, gbMesh3D};

            Reg3DCustom::performRegistrationSingleMesh(
                organs, liverMesh3D, mergedIndices,
                screenMesh, OrbitCam.cameraPos,
                gGridWidth, gGridHeight(), 15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);

            std::cout << "=== Registration Complete ===" << std::endl;
        }
        break;

    case GLFW_KEY_W:
        if (currentMainMode == REGISTRATION_MODE) {
            std::cout << "\n=== Multi-Start FGR with Adaptive Radius ===\n" << std::endl;

            poseAutoSaveBeforeRegistration();
            resetRegistrationState();

            Reg3D::BVHTree msFgrBvh;
            msFgrBvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);

            Reg3D::RaycastClusterer clusterer(msFgrBvh);
            auto clusteringResult = clusterer.performClustering(
                liverMesh3D->mVertices, liverMesh3D->mIndices);

            Reg3DCustom::NoOpen3DRegistration reg;
            auto targetCloud = reg.extractFrontFacePoints(
                *screenMesh, gGridWidth, gGridHeight(), gDepthScale);

            if (targetCloud->size() < 100) {
                std::cerr << "[W] Not enough target points" << std::endl;
                break;
            }

            struct ClusterFGRResult {
                int clusterId;
                float fgrFitness;
                float fgrRmse;
                std::vector<size_t> vertexIndices;
                float adaptiveVoxel;
            };

            std::vector<ClusterFGRResult> fgrResults;

            for (const auto& cluster : clusteringResult.clusters) {
                if (cluster.visibleVertexIndices.size() < 50) continue;

                glm::vec3 bboxMin(FLT_MAX), bboxMax(-FLT_MAX);
                for (const auto& v : cluster.visibleVertices) {
                    bboxMin = glm::min(bboxMin, v);
                    bboxMax = glm::max(bboxMax, v);
                }
                float bboxDiag = glm::length(bboxMax - bboxMin);
                float adaptiveVoxel = glm::clamp(bboxDiag * 0.05f, 0.3f, 1.5f);

                auto sourceCloud = std::make_shared<Reg3DCustom::PointCloud>();
                for (int idx : cluster.visibleVertexIndices) {
                    size_t i = static_cast<size_t>(idx);
                    if (i * 3 + 2 < liverMesh3D->mVertices.size()) {
                        glm::vec3 pos(liverMesh3D->mVertices[i*3],
                                      liverMesh3D->mVertices[i*3+1],
                                      liverMesh3D->mVertices[i*3+2]);
                        if (!liverMesh3D->mNormals.empty() && i*3+2 < liverMesh3D->mNormals.size()) {
                            glm::vec3 nrm(liverMesh3D->mNormals[i*3],
                                          liverMesh3D->mNormals[i*3+1],
                                          liverMesh3D->mNormals[i*3+2]);
                            sourceCloud->addPointWithNormal(pos, nrm);
                        } else {
                            sourceCloud->addPoint(pos);
                        }
                    }
                }

                auto sourceDown = reg.preprocess(sourceCloud, adaptiveVoxel, true);
                auto targetDown = reg.preprocess(targetCloud, adaptiveVoxel, false);

                if (sourceDown->size() < 10 || targetDown->size() < 10) continue;

                auto sourceFpfh = reg.computeFPFH(sourceDown, adaptiveVoxel);
                auto targetFpfh = reg.computeFPFH(targetDown, adaptiveVoxel);

                auto fgrResult = reg.fastGlobalRegistration(
                    sourceDown, targetDown, sourceFpfh, targetFpfh, adaptiveVoxel);

                std::cout << "[W] Cluster " << cluster.clusterId
                          << " bbox=" << bboxDiag
                          << " voxel=" << adaptiveVoxel
                          << " fitness=" << fgrResult.fitness
                          << " rmse=" << fgrResult.inlier_rmse << std::endl;

                fgrResults.push_back({
                    cluster.clusterId,
                    fgrResult.fitness,
                    fgrResult.inlier_rmse,
                    std::vector<size_t>(cluster.visibleVertexIndices.begin(),
                                        cluster.visibleVertexIndices.end()),
                    adaptiveVoxel
                });
            }

            if (fgrResults.empty()) {
                std::cerr << "[W] No valid FGR results" << std::endl;
                break;
            }

            std::sort(fgrResults.begin(), fgrResults.end(),
                      [](const ClusterFGRResult& a, const ClusterFGRResult& b) {
                          return a.fgrFitness > b.fgrFitness;
                      });

            int topN = std::min(3, static_cast<int>(fgrResults.size()));
            std::cout << "\n[W] Running full registration on top " << topN << " clusters..." << std::endl;

            float bestCompRmse = FLT_MAX;
            int bestClusterId = -1;

            std::vector<std::vector<float>> savedVertices(6), savedNormals(6);
            std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                              tumorMesh3D, segmentMesh3D, gbMesh3D};
            for (int m = 0; m < 6; m++) {
                savedVertices[m] = organs[m]->mVertices;
                savedNormals[m]  = organs[m]->mNormals;
            }

            std::vector<float> bestVertices0, bestNormals0;

            for (int i = 0; i < topN; i++) {
                for (int m = 0; m < 6; m++) {
                    organs[m]->mVertices = savedVertices[m];
                    organs[m]->mNormals  = savedNormals[m];
                }

                std::cout << "\n[W] Candidate " << (i+1) << "/" << topN
                          << " (cluster " << fgrResults[i].clusterId
                          << ", fitness=" << fgrResults[i].fgrFitness << ")" << std::endl;

                Reg3DCustom::performRegistrationSingleMesh(
                    organs, liverMesh3D, fgrResults[i].vertexIndices,
                    screenMesh, OrbitCam.cameraPos,
                    gGridWidth, gGridHeight(), 15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);

                computeUnifiedMetrics();
                float compRmse = registrationHandle.compRmse;

                std::cout << "[W] Candidate " << (i+1) << " CompRMSE=" << compRmse << std::endl;

                if (compRmse < bestCompRmse) {
                    bestCompRmse = compRmse;
                    bestClusterId = fgrResults[i].clusterId;
                    bestVertices0 = organs[0]->mVertices;
                    bestNormals0  = organs[0]->mNormals;
                    for (int m = 0; m < 6; m++) {
                        savedVertices[m] = organs[m]->mVertices;
                        savedNormals[m]  = organs[m]->mNormals;
                    }
                }
            }

            for (int m = 0; m < 6; m++) {
                organs[m]->mVertices = savedVertices[m];
                organs[m]->mNormals  = savedNormals[m];
            }

            gUIManager.state.regMethod = 1;
            registrationHandle.state = RegistrationData::REGISTERED;
            registrationHandle.useRegistration = true;
            computeUnifiedMetrics();
            poseSaveToLibrary();

            std::cout << "\n[W] Best cluster: " << bestClusterId
                      << " CompRMSE=" << bestCompRmse << std::endl;
            std::cout << "=== Multi-Start FGR Complete ===" << std::endl;
        }
        break;

    case GLFW_KEY_F:
        if (currentMainMode == REGISTRATION_MODE) {
            openImageFilePicker();
        }
        break;

    case GLFW_KEY_V:
        if (currentMainMode == REGISTRATION_MODE && (mode & GLFW_MOD_SHIFT)) {
            std::cout << "\n=== BIPOP-CMA-ES Multi-Start (Shift+V) ===" << std::endl;
            if (registrationHandle.compRmse == 0.0f) {
                std::cerr << "[Shift+V] No registration yet. Run HemiAuto first." << std::endl;
                break;
            }
            g_stepStartTime = std::chrono::steady_clock::now();
            g_sessionBipopN++;
            gUIManager.state.regMethod = 3;
            poseAutoSaveBeforeRegistration();
            auto organs = getOrganList();
            computeUnifiedMetrics();
            float rmse_before = registrationHandle.compRmse;
            std::cout << "[Shift+V] Current compRMSE: " << rmse_before << std::endl;

            /* 現在の頂点をスナップショット */
            std::vector<std::vector<GLfloat>> start_v(organs.size());
            std::vector<std::vector<GLfloat>> start_n(organs.size());
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i]) {
                    start_v[i] = organs[i]->mVertices;
                    start_n[i] = organs[i]->mNormals;
                }
            }

            float best_rmse = rmse_before;
            std::vector<std::vector<GLfloat>> best_v = start_v;
            std::vector<std::vector<GLfloat>> best_n = start_n;

            const int N_STARTS = 10;
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

            for (int run = 0; run < N_STARTS; run++) {
                /* 毎回start_vに戻す */
                for (size_t i = 0; i < organs.size(); i++) {
                    if (organs[i]) {
                        organs[i]->mVertices = start_v[i];
                        organs[i]->mNormals  = start_n[i];
                        setUp(*organs[i]);
                    }
                }

                CmaesRefine::Params p;
                p.verbose        = true;
                p.log_every      = 100;
                p.save_debug_jpg = false;
                p.maxgen         = 300;
                p.tx_range = 1.0f; p.ty_range = 1.0f; p.tz_range = 1.0f;
                p.rx_range = 20.0f; p.ry_range = 20.0f; p.rz_range = 20.0f;
                p.scale_lo = 0.85f; p.scale_hi = 1.15f;

                float tx_perturb = 0.0f, ty_perturb = 0.0f, tz_perturb = 0.0f;
                float rx_perturb = 0.0f, ry_perturb = 0.0f, rz_perturb = 0.0f;
                float sc_perturb = 1.0f;
                std::string regime;

                if (run == 0) {
                    /* Run 0: 摂動なし・sigma0小（精密ベースライン） */
                    p.sigma0 = 0.2;
                    regime = "Baseline";
                } else if (run <= 4) {
                    /* Run 1-4: Regime 2（小sigma0・小摂動・局所探索） */
                    p.sigma0 = 0.05f + dist01(rng) * 0.25f; /* 0.05〜0.30 */
                    tx_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
                    ty_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
                    tz_perturb = (dist01(rng)*2.0f-1.0f) * 0.5f;
                    rx_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
                    ry_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
                    rz_perturb = (dist01(rng)*2.0f-1.0f) * 10.0f;
                    sc_perturb = 0.95f + dist01(rng) * 0.10f; /* 0.95〜1.05 */
                    regime = "Regime2(local)";
                } else {
                    /* Run 5-9: Regime 1（大sigma0・大摂動・広域探索） */
                    p.sigma0 = 0.30f + dist01(rng) * 0.50f; /* 0.30〜0.80 */
                    tx_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
                    ty_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
                    tz_perturb = (dist01(rng)*2.0f-1.0f) * 1.5f;
                    rx_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
                    ry_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
                    rz_perturb = (dist01(rng)*2.0f-1.0f) * 30.0f;
                    sc_perturb = 0.90f + dist01(rng) * 0.20f; /* 0.90〜1.10 */
                    regime = "Regime1(global)";
                }

                /* 初期摂動を適用 */
                if (run > 0) {
                    CmaesRefine::applyIncrementalSRT(organs,
                                                     tx_perturb, ty_perturb, tz_perturb,
                                                     rx_perturb, ry_perturb, rz_perturb,
                                                     sc_perturb);
                    for (size_t i = 0; i < organs.size(); i++)
                        if (organs[i]) setUp(*organs[i]);
                }

                std::cout << "[Shift+V] Run " << (run+1) << "/" << N_STARTS
                          << "  " << regime
                          << "  sigma0=" << std::fixed << std::setprecision(2) << p.sigma0
                          << std::endl;

                CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                                         gGridWidth, gGridHeight(), gDepthScale, p);
                computeUnifiedMetrics();
                float rmse_run = registrationHandle.compRmse;
                std::cout << "[Shift+V] Run " << (run+1)
                          << " compRMSE=" << std::setprecision(6) << rmse_run
                          << (r.improved ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;

                if (rmse_run < best_rmse) {
                    best_rmse = rmse_run;
                    for (size_t i = 0; i < organs.size(); i++) {
                        if (organs[i]) {
                            best_v[i] = organs[i]->mVertices;
                            best_n[i] = organs[i]->mNormals;
                        }
                    }
                }
            }

            /* ベスト姿勢を適用 */
            for (size_t i = 0; i < organs.size(); i++) {
                if (organs[i]) {
                    organs[i]->mVertices = best_v[i];
                    organs[i]->mNormals  = best_n[i];
                    setUp(*organs[i]);
                }
            }
            computeUnifiedMetrics();
            std::cout << "[Shift+V] Best: " << rmse_before
                      << " -> " << best_rmse
                      << (best_rmse < rmse_before ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;
            poseSaveToLibrary();

            /* Shift+T (Ctrl+Shift+E) 軌道検索の端点として最終姿勢を保存 */
            if (!g_initOrganVertices.empty() && liverMesh3D &&
                !g_initOrganVertices[0].empty() &&
                g_initOrganVertices[0].size() == liverMesh3D->mVertices.size()) {
                gShiftV_lastTransform = PoseLibrary::computeTransformFromLiver(
                    g_initOrganVertices[0], liverMesh3D->mVertices);
                gShiftV_lastValid = true;
                std::cout << "[Shift+T] BIPOP endpoint saved (gShiftV_lastTransform)" << std::endl;
            }
        } else {
            g_showClusterVisualization = !g_showClusterVisualization;
            std::cout << "Cluster visualization: "
                      << (g_showClusterVisualization ? "ON" : "OFF") << std::endl;
        }
        break;

    case GLFW_KEY_G:
        if(currentMainMode == REGISTRATION_MODE){
            std::cout << "\n============================================" << std::endl;
            std::cout << "  Starting Umeyama Registration Mode" << std::endl;
            std::cout << "============================================\n" << std::endl;

            isDragging = false;
            hit_index = -1;

            if (splitScreenMode) {
                splitScreenMode = false;
                std::cout << "Turning off previous split screen mode..." << std::endl;
            }

            registrationHandle.reset();
            registrationHandle.targetPointCount = 5;
            registrationHandle.state = RegistrationData::SELECTING_BOARD_POINTS;
            registrationHandle.useRegistration = false;

            std::cout << "Registration handle reset complete" << std::endl;

            OrbitCam.resetToInitialState();
            OrbitCam.cx = gWindowWidth / 2.0f;
            OrbitCam.cy = gWindowHeight / 2.0f;

            std::cout << "Main camera reset complete" << std::endl;

            OrbitCamLeft_Target = OrbitCam;
            OrbitCamRight_Screen = OrbitCam;

            OrbitCamLeft_Target.gRadius = OrbitCam.InitialRadius * 1.0f;
            OrbitCamLeft_Target.currentTarget = TARGET_LIVER;
            OrbitCamLeft_Target.cx = (gWindowWidth / 2) / 2.0f;
            OrbitCamLeft_Target.cy = gWindowHeight / 2.0f;

            OrbitCamRight_Screen.gRadius = OrbitCam.InitialRadius * 2.0f;
            OrbitCamRight_Screen.currentTarget = TARGET_TEXTURE;
            OrbitCamRight_Screen.cx = (gWindowWidth / 2) / 2.0f;
            OrbitCamRight_Screen.cy = gWindowHeight / 2.0f;

            std::cout << "Split screen cameras configured" << std::endl;
            std::cout << "  Left camera radius: " << OrbitCamLeft_Target.gRadius << std::endl;
            std::cout << "  Right camera radius: " << OrbitCamRight_Screen.gRadius << std::endl;

            splitScreenMode = true;

            std::cout << "\n=== Umeyama Registration Mode Started (5 points) ===" << std::endl;
            std::cout << "1. Select 5 points on texture board (RIGHT screen)" << std::endl;
            std::cout << "2. Then select 5 corresponding points on liver (LEFT screen)" << std::endl;
            std::cout << "3. Press T to execute registration\n" << std::endl;
            std::cout << "=== Split Screen Mode: ON ===" << std::endl;
        }
        break;

    case GLFW_KEY_T:
        if(currentMainMode == REGISTRATION_MODE){
            if (registrationHandle.canRegister()) {
                poseAutoSaveBeforeRegistration();
                std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D, tumorMesh3D, segmentMesh3D, gbMesh3D};
                performRegistrationUmeyama(registrationHandle, organs);
                computeUnifiedMetrics();
                poseSaveToLibrary();
                std::cout << "Registration complete: " << std::endl;
            }

            splitScreenMode = false;
            OrbitCam = OrbitCamRight_Screen;
            OrbitCam.gRadius = OrbitCam.InitialRadius;
            OrbitCam.cx = gWindowWidth / 2.0f;
            OrbitCam.cy = gWindowHeight / 2.0f;

            std::cout << "=== Split Screen Mode OFF ===" << std::endl;
        }
        break;

    case GLFW_KEY_J:
        if(currentMainMode == REGISTRATION_MODE){
            if(registrationHandle.state == RegistrationData::SELECTING_BOARD_POINTS ||
                registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS){
                registrationHandle.reset();
                std::cout << "Registration reset" << std::endl;
                registrationHandle.targetPointCount = 5;
                registrationHandle.state = RegistrationData::SELECTING_BOARD_POINTS;
                std::cout << "=== Registration Mode Started (5 points) ===" << std::endl;
                std::cout << "Select 5 points on texture board" << std::endl;
            } else {
                registrationHandle.reset();
                std::cout << "Registration reset" << std::endl;
            }
        }
        break;

    case GLFW_KEY_H:
        if(currentMainMode == DEFORM_MODE) {
            if(deformHandlPlace.state  != DeformHandlPlaceData::DEFORM_MODE){
                multiBody->setRigidMode(true);
                multiBody->initPhysics();
                deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
                std::cout << "=== HandlePlace Mode Started (max "
                          << SoftBody::MAX_HANDLE_GROUPS << " groups) ===" << std::endl;
                std::cout << "Current groups: " << multiBody->handleGroups.size()
                          << "/" << SoftBody::MAX_HANDLE_GROUPS << std::endl;
            }

            if(deformHandlPlace.state  == DeformHandlPlaceData::DEFORM_MODE){
                if (multiBody->handleGroups.size() < SoftBody::MAX_HANDLE_GROUPS){
                    multiBody->setRigidMode(true);
                    deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
                } else {
                    deformHandlPlace.reset();
                    if (multiBody) {
                        multiBody->fullReset();
                    }
                    std::cout << "Complete reset performed" << std::endl;

                    multiBody->setRigidMode(true);
                    multiBody->initPhysics();
                    deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
                    std::cout << "=== HandlePlace Mode Started (max "
                              << SoftBody::MAX_HANDLE_GROUPS << " groups) ===" << std::endl;
                    std::cout << "Current groups: " << multiBody->handleGroups.size()
                              << "/" << SoftBody::MAX_HANDLE_GROUPS << std::endl;
                }
            }
        }

        break;

    case GLFW_KEY_M:
        liverMesh3D->exportObjFile(Reg_TARGET_FILE_PATH);
        portalMesh3D->exportObjFile(Reg_PORTAL_FILE_PATH);
        veinMesh3D->exportObjFile(Reg_VEIN_FILE_PATH);
        tumorMesh3D->exportObjFile(Reg_TUMOR_FILE_PATH);
        segmentMesh3D->exportObjFile(Reg_SEGMENT_FILE_PATH);
        gbMesh3D->exportObjFile(Reg_GB_FILE_PATH);

        currentMainMode = DEFORM_MODE;
        break;

        if(currentMainMode == REGISTRATION_MODE) {
            currentMainMode = DEFORM_MODE;
            std::cout << "=== Switched to DEFORM_MODE ===" << std::endl;
        } else {
            currentMainMode = REGISTRATION_MODE;
            std::cout << "=== Switched to REGISTRATION_MODE ===" << std::endl;
        }
        break;

    case GLFW_KEY_D:
        if (currentMainMode == REGISTRATION_MODE && (mode & GLFW_MOD_SHIFT)) {
            /* ターゲット輪郭が depth に依存するか検証するデバッグダンプ (Shift+D)。
               depthScale を変えた前後で Shift+D を押してログを比較することで、
               Shift+E の目的関数 target (g_boundaryDistMap) が 2D マスクベースで
               depth 非依存であること、一方 Key E や UI 表示用の target
               (g_targetPoints / g_cluster2Points) は screenMesh の 3D 頂点ベースで
               depth 依存であることを確認できる。 */
            std::cout << "\n========== [Shift+D Debug] Target contour dump ==========" << std::endl;
            std::cout << "[Debug] depthScale = " << gDepthScale << std::endl;

            /* ダンプ前に強制的に computeUnifiedMetrics を呼ぶ。
               onDepthScaleChanged は g_showClusterVisualization && REGISTERED の
               条件でしか computeUnifiedMetrics を呼ばないため、スライダーで
               depthScale を変えただけでは g_targetPoints / g_cluster2Points が
               古い screenMesh 状態のまま残る。ダンプ時点の screenMesh 状態を
               正確に反映させるため、ここで明示的に再計算する。 */
            std::cout << "[Debug] (forcing computeUnifiedMetrics to refresh g_targetPoints)" << std::endl;
            computeUnifiedMetrics();

            /* --- (1) 2D mask target: g_boundaryDistMap ---
               これが Shift+E の computeSilhouette2DObjective で使われる本来の target。
               PNG ファイル (segmentation_mask.png) から構築されており、
               depthScale の変更では再構築されない。つまり depth-INdependent のはず。 */
            std::cout << "\n[Debug] --- (1) g_boundaryDistMap (2D mask, expected depth-INdependent) ---" << std::endl;
            if (!g_boundaryDistMap.valid) {
                std::cout << "[Debug]   NOT VALID (mask not loaded)" << std::endl;
            } else {
                int mw = g_boundaryDistMap.width;
                int mh = g_boundaryDistMap.height;
                int interiorCount = 0;
                int boundaryCount = 0;
                double sumX = 0.0, sumY = 0.0;
                for (int y = 0; y < mh; y++) {
                    for (int x = 0; x < mw; x++) {
                        float bd = g_boundaryDistMap.data[y * mw + x];
                        if (bd < 9000.0f) {
                            interiorCount++;
                            sumX += x;
                            sumY += y;
                        }
                        if (bd < 1.5f) boundaryCount++;
                    }
                }
                double cx = interiorCount > 0 ? sumX / interiorCount : 0.0;
                double cy = interiorCount > 0 ? sumY / interiorCount : 0.0;
                std::cout << "[Debug]   map size: " << mw << " x " << mh << std::endl;
                std::cout << "[Debug]   interior pixel count (bd<9000): " << interiorCount << std::endl;
                std::cout << "[Debug]   boundary pixel count (bd<1.5):  " << boundaryCount << std::endl;
                std::cout << "[Debug]   interior centroid (px): (" << cx << ", " << cy << ")" << std::endl;
            }

            /* --- (2) 3D boundary target: g_targetPoints ---
               computeUnifiedMetrics で screenMesh の 3D 頂点から構築される。
               screenMesh は depthScale で z 座標が変わるため、z 分布も変わる。
               ただし xy 座標は depth 非依存なので変わらないはず。 */
            std::cout << "\n[Debug] --- (2) g_targetPoints (3D boundary, expected depth-dependent in Z) ---" << std::endl;
            std::cout << "[Debug]   count = " << g_targetPoints.size() << std::endl;
            if (!g_targetPoints.empty()) {
                float xmin=FLT_MAX, xmax=-FLT_MAX;
                float ymin=FLT_MAX, ymax=-FLT_MAX;
                float zmin=FLT_MAX, zmax=-FLT_MAX;
                double sumX=0, sumY=0, sumZ=0;
                for (const auto& p : g_targetPoints) {
                    xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
                    ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
                    zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
                    sumX += p.x; sumY += p.y; sumZ += p.z;
                }
                size_t n = g_targetPoints.size();
                std::cout << "[Debug]   x range: [" << xmin << ", " << xmax
                          << "]  (depth-INdependent, should be stable)" << std::endl;
                std::cout << "[Debug]   y range: [" << ymin << ", " << ymax
                          << "]  (depth-INdependent, should be stable)" << std::endl;
                std::cout << "[Debug]   z range: [" << zmin << ", " << zmax
                          << "]  (depth-dependent, should scale with depthScale)" << std::endl;
                std::cout << "[Debug]   centroid: (" << sumX/n << ", " << sumY/n << ", " << sumZ/n << ")" << std::endl;
            }

            /* --- (3) 3D interior target: g_cluster2Points ---
               同様に screenMesh の 3D 内部頂点。画像の青点群。 */
            std::cout << "\n[Debug] --- (3) g_cluster2Points (3D interior, expected depth-dependent in Z) ---" << std::endl;
            std::cout << "[Debug]   count = " << g_cluster2Points.size() << std::endl;
            if (!g_cluster2Points.empty()) {
                float zmin=FLT_MAX, zmax=-FLT_MAX;
                double sumZ=0;
                for (const auto& p : g_cluster2Points) {
                    zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
                    sumZ += p.z;
                }
                std::cout << "[Debug]   z range: [" << zmin << ", " << zmax << "]" << std::endl;
                std::cout << "[Debug]   z mean: " << sumZ / g_cluster2Points.size() << std::endl;
            }

            /* --- (4) 現在の IoU を直接計算してログに出す ---
               g_boundaryDistMap が depth 非依存なら、IoU も (肝臓姿勢が同じ限り)
               depthScale の変更で変化しないはず。 */
            std::cout << "\n[Debug] --- (4) Current 2D IoU (expected depth-INdependent) ---" << std::endl;
            if (liverMesh3D && g_boundaryDistMap.valid) {
                CmaesRefine::SilBVH dummy;
                float fval = CmaesRefine::computeSilhouette2DObjective(
                    liverMesh3D, dummy, view, projection, 4);
                std::cout << "[Debug]   1-IoU = " << fval
                          << "   IoU = " << (1.0f - fval) << std::endl;
            } else {
                std::cout << "[Debug]   skipped (liver or mask not ready)" << std::endl;
            }

            std::cout << "========== [Shift+D Debug] END ==========\n" << std::endl;
        } else if (currentMainMode == DEFORM_MODE) {
            multiBody->setRigidMode(false);
            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
            std::cout << "Mode: DEFORM MODE" << std::endl;
        }

        break;

    case GLFW_KEY_R:
        if(currentMainMode == DEFORM_MODE) {
            deformHandlPlace.state = DeformHandlPlaceData::RIGID_MODE;
            multiBody->setRigidMode(true);
            std::cout << "Mode: RIGID MODE" << std::endl;
        }
        if (!cutterMesh) std::cerr << "[Warning] cutterMesh is not initialized, plane cut will not work" << std::endl;

        break;

    case GLFW_KEY_P:
        if ((mode & GLFW_MOD_SHIFT) && (mode & GLFW_MOD_CONTROL)) {
            if (!screenMesh || screenMesh->mVertices.empty()) {
                std::cout << "[Export] No screenMesh" << std::endl;
                break;
            }
            int gw = gGridWidth;
            int gh = gw * screenMesh->loadedImageHeight / screenMesh->loadedImageWidth;
            size_t nFront = (size_t)(gw + 1) * (gh + 1);
            int imgW = screenMesh->loadedImageWidth;
            int imgH = screenMesh->loadedImageHeight;
            std::vector<float> depths;
            if (gPinMode == 2) {
                std::string maskPath = std::string(DEPTH_OUTPUT_PATH) + "segmentation_mask.png";
                std::vector<unsigned char> maskData;
                int mW = 0, mH = 0;
                if (loadMaskPNG(maskPath, maskData, mW, mH)) {
                    depths = screenMesh->calculateBlockDepthMetric(gw, gh, maskData, mW, mH, 0.9f);
                } else {
                    depths = screenMesh->calculateBlockDepthRaw(gw, gh, 0.9f);
                }
            } else {
                depths = screenMesh->calculateNormalizedDepth(gw, gh, 0.99f, 0.9f);
            }
            const float halfThk = 0.025f;

            std::cout << "\n[Export] === VERIFY THEORETICAL VALUES ===" << std::endl;
            const char* modeName = (gPinMode == 0) ? "relief"
                                 : (gPinMode == 1) ? "diff_pinhole"
                                                   : "pure_pinhole";
            std::cout << "[Export] gDepthScale=" << gDepthScale
                      << " gPinMode=" << modeName
                      << " gPinholeBaseScale=" << gPinholeBaseScale
                      << " meshScale=" << gMeshScale << std::endl;
            if (gLatestIntrinsics.valid) {
                std::cout << "[Export] intrinsics fx=" << gLatestIntrinsics.fx
                          << " fy=" << gLatestIntrinsics.fy
                          << " cx=" << gLatestIntrinsics.cx
                          << " cy=" << gLatestIntrinsics.cy
                          << " w=" << gLatestIntrinsics.width
                          << " h=" << gLatestIntrinsics.height << std::endl;
            }
            float planeW = (imgW > 0 && imgH > 0) ? (float)imgW / (float)imgH : (float)gw / gh;
            size_t testIdx[] = {0, (size_t)(gw), nFront / 2, nFront - 1, nFront - (size_t)(gw + 1)};
            const char* testName[] = {"TL", "TR", "CTR", "BR", "BL"};
            float delta = gDepthScale - gPinholeBaseScale;
            float pureScale = (gPinMode == 2 && gLatestIntrinsics.valid)
                              ? (planeW * gLatestIntrinsics.fx / (float)imgW) : 1.0f;
            bool allOk = true;
            for (int k = 0; k < 5; ++k) {
                size_t i = testIdx[k];
                int gx = (int)(i % (gw + 1));
                int gy = (int)(i / (gw + 1));
                float u = (float)gx / gw;
                float v = (float)gy / gh;
                float depth = depths[i];
                float zRaw = halfThk + depth * gDepthScale;
                float expZ = zRaw * gMeshScale;
                float expX, expY;
                if (gPinMode == 1 && gLatestIntrinsics.valid) {
                    float px = u * (float)imgW;
                    float py = v * (float)imgH;
                    float ZDelta = depth * delta;
                    expX = ((u - 0.5f) * planeW + (px - gLatestIntrinsics.cx) * ZDelta / gLatestIntrinsics.fx) * gMeshScale;
                    expY = ((0.5f - v) * 1.0f - (py - gLatestIntrinsics.cy) * ZDelta / gLatestIntrinsics.fy) * gMeshScale;
                } else if (gPinMode == 2 && gLatestIntrinsics.valid) {
                    float px = u * (float)imgW;
                    float py = v * (float)imgH;
                    float Zpure = depth * gDepthScale;
                    if (Zpure < 1e-6f) Zpure = 1e-6f;
                    expX =  (px - gLatestIntrinsics.cx) * Zpure / gLatestIntrinsics.fx * pureScale * gMeshScale;
                    expY = -(py - gLatestIntrinsics.cy) * Zpure / gLatestIntrinsics.fy * pureScale * gMeshScale;
                    expZ = Zpure * gMeshScale;
                } else {
                    expX = (u - 0.5f) * planeW * gMeshScale;
                    expY = (0.5f - v) * 1.0f  * gMeshScale;
                }
                float ax = screenMesh->mVertices[i*3+0];
                float ay = screenMesh->mVertices[i*3+1];
                float az = screenMesh->mVertices[i*3+2];
                float ex = expX - ax, ey = expY - ay, ez = expZ - az;
                bool ok = (std::abs(ex) < 1e-3f && std::abs(ey) < 1e-3f && std::abs(ez) < 1e-3f);
                if (!ok) allOk = false;
                std::cout << "  [" << testName[k] << "] i=" << i
                          << " uv=(" << u << "," << v << ") d=" << depth << std::endl;
                std::cout << "      actual = (" << ax << ", " << ay << ", " << az << ")" << std::endl;
                std::cout << "      expect = (" << expX << ", " << expY << ", " << expZ << ")" << std::endl;
                std::cout << "      diff   = (" << ex << ", " << ey << ", " << ez << ") "
                          << (ok ? "OK" : "MISMATCH") << std::endl;
            }
            std::cout << "[Export] Verification: " << (allOk ? "ALL OK" : "MISMATCH") << std::endl;

            std::string basePath = std::string(DEPTH_OUTPUT_PATH) + "screenMesh_" + modeName;
            std::string objPath = basePath + ".obj";
            std::string mtlPath = basePath + ".mtl";
            std::string plyPath = basePath + ".ply";
            std::string mtlName = std::string("screenMesh_") + modeName;
            {
                std::ofstream f(objPath);
                if (!f.is_open()) { std::cerr << "[Export] Cannot open " << objPath << std::endl; break; }
                f << "# screenMesh export, " << modeName << "\n";
                f << "# depthScale=" << gDepthScale << " base=" << gPinholeBaseScale << "\n";
                f << "mtllib " << mtlName << ".mtl\n";
                for (size_t i = 0; i < nFront; ++i) {
                    f << "v " << screenMesh->mVertices[i*3+0] << " "
                              << screenMesh->mVertices[i*3+1] << " "
                              << screenMesh->mVertices[i*3+2] << "\n";
                }
                for (int y = 0; y <= gh; ++y) {
                    for (int x = 0; x <= gw; ++x) {
                        float u = (float)x / gw;
                        float v = (float)y / gh;
                        f << "vt " << u << " " << (1.0f - v) << "\n";
                    }
                }
                f << "usemtl " << mtlName << "\n";
                for (int y = 0; y < gh; ++y) {
                    for (int x = 0; x < gw; ++x) {
                        int a = y * (gw + 1) + x + 1;
                        int b = a + 1;
                        int c = (y + 1) * (gw + 1) + x + 1;
                        int d = c + 1;
                        f << "f " << a << "/" << a << " "
                                   << c << "/" << c << " "
                                   << b << "/" << b << "\n";
                        f << "f " << b << "/" << b << " "
                                   << c << "/" << c << " "
                                   << d << "/" << d << "\n";
                    }
                }
                f.close();
                std::cout << "[Export] OBJ saved: " << objPath << std::endl;
            }
            {
                std::ofstream f(mtlPath);
                if (!f.is_open()) { std::cerr << "[Export] Cannot open " << mtlPath << std::endl; break; }
                f << "newmtl " << mtlName << "\n";
                f << "Ka 1.0 1.0 1.0\n";
                f << "Kd 1.0 1.0 1.0\n";
                f << "Ks 0.0 0.0 0.0\n";
                f << "map_Kd original.jpg\n";
                f.close();
                std::cout << "[Export] MTL saved: " << mtlPath << std::endl;
            }
            {
                std::ofstream f(plyPath);
                if (!f.is_open()) { std::cerr << "[Export] Cannot open " << plyPath << std::endl; break; }
                int nFaces = gw * gh * 2;
                std::string origPath = std::string(DEPTH_OUTPUT_PATH) + "original.jpg";
                int imgChan = 0, imgWLoaded = 0, imgHLoaded = 0;
                unsigned char* imgData = stbi_load(origPath.c_str(), &imgWLoaded, &imgHLoaded, &imgChan, 3);
                bool hasColor = (imgData != nullptr && imgWLoaded > 0 && imgHLoaded > 0);
                f << "ply\nformat ascii 1.0\n";
                f << "element vertex " << nFront << "\n";
                f << "property float x\nproperty float y\nproperty float z\n";
                if (hasColor) {
                    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
                }
                f << "element face " << nFaces << "\n";
                f << "property list uchar int vertex_indices\n";
                f << "end_header\n";
                for (int gy = 0; gy <= gh; ++gy) {
                    for (int gx = 0; gx <= gw; ++gx) {
                        size_t i = (size_t)gy * (gw + 1) + gx;
                        f << screenMesh->mVertices[i*3+0] << " "
                          << screenMesh->mVertices[i*3+1] << " "
                          << screenMesh->mVertices[i*3+2];
                        if (hasColor) {
                            float u = (float)gx / gw;
                            float v = (float)gy / gh;
                            int px = std::min((int)(u * imgWLoaded), imgWLoaded - 1);
                            int py = std::min((int)(v * imgHLoaded), imgHLoaded - 1);
                            unsigned char R = imgData[(py * imgWLoaded + px) * 3 + 0];
                            unsigned char G = imgData[(py * imgWLoaded + px) * 3 + 1];
                            unsigned char B = imgData[(py * imgWLoaded + px) * 3 + 2];
                            f << " " << (int)R << " " << (int)G << " " << (int)B;
                        }
                        f << "\n";
                    }
                }
                for (int y = 0; y < gh; ++y) {
                    for (int x = 0; x < gw; ++x) {
                        int a = y * (gw + 1) + x;
                        int b = a + 1;
                        int c = (y + 1) * (gw + 1) + x;
                        int d = c + 1;
                        f << "3 " << a << " " << c << " " << b << "\n";
                        f << "3 " << b << " " << c << " " << d << "\n";
                    }
                }
                f.close();
                if (imgData) stbi_image_free(imgData);
                std::cout << "[Export] PLY saved: " << plyPath
                          << (hasColor ? " (with vertex colors)" : " (no color)") << std::endl;
            }
            break;
        }
        if (mode & GLFW_MOD_SHIFT) {
            if (currentMainMode == DEFORM_MODE) {
                const auto& presets = AutoDeform::getPresets();
                gAutoDeformPresetIdx = (gAutoDeformPresetIdx + 1) % static_cast<int>(presets.size());
                const auto& p = presets[gAutoDeformPresetIdx];
                std::cout << "\n[AutoDeform Preset] " << p.name
                          << "  K_fix=" << p.K_fix << " K_move=" << p.K_move
                          << " rFix=" << p.rFixScale << " rMove=" << p.rMoveScale
                          << " fixC=" << p.fixCompliance << " moveC=" << p.moveCompliance
                          << std::endl;
            } else if (currentMainMode == REGISTRATION_MODE) {
                /* Shift+P = run MICCAI-WS Shift+P protocol batch for
                 * the currently loaded image. Blocks the UI for ~50 s.
                 * Logs to protocol_YYYYMMDD_HHMMSS.csv/.txt. */
                std::string imageName = g_currentOrientLabel.empty()
                                        ? std::string("session")
                                        : g_currentOrientLabel;
                runProtocolBatch(imageName);
            }
        } else if (currentMainMode == DEFORM_MODE) {
            multiBody->setRigidMode(true);
            deformHandlPlace.state = DeformHandlPlaceData::PLANECUT_MODE;
            std::cout << "Mode: PLANECUT_MODE" << std::endl;
        }
        break;

    case GLFW_KEY_C:
        if(currentMainMode == REGISTRATION_MODE){
            if (mode & GLFW_MOD_SHIFT) {
                /* Shift+C = CMA-ES only（現在の姿勢からそのまま精密化） */
                std::cout << "\n=== CMA-ES Only (Shift+C) ===" << std::endl;
                if (registrationHandle.compRmse == 0.0f) {
                    std::cerr << "[Shift+C] No registration yet. Run HemiAuto first." << std::endl;
                    break;
                }
                poseAutoSaveBeforeRegistration();
                auto organs = getOrganList();
                computeUnifiedMetrics();
                float rmse_before = registrationHandle.compRmse;
                std::cout << "[Shift+C] Current compRMSE: " << rmse_before << std::endl;

                CmaesRefine::Params p;
                p.verbose   = true;
                p.log_every = 30;
                p.save_debug_jpg = false;
                p.tx_range = 1.0f;
                p.ty_range = 1.0f;
                p.tz_range = 1.0f;
                p.rx_range = 20.0f;
                p.ry_range = 20.0f;
                p.rz_range = 20.0f;
                p.scale_lo = 0.85f;
                p.scale_hi = 1.15f;
                CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                                         gGridWidth, gGridHeight(), gDepthScale, p);
                computeUnifiedMetrics();
                std::cout << "[Shift+C] Result: " << rmse_before
                          << " -> " << registrationHandle.compRmse
                          << (r.improved ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;
                poseSaveToLibrary();
            } else {
                /* Key C = HemiAuto + CMA-ES */
                std::cout << "\n=== HemiAuto + CMA-ES Mode (Key C) ===" << std::endl;
                gUIManager.state.regMethod = 1;
                poseAutoSaveBeforeRegistration();
                resetRegistrationState();

                Reg3D::BVHTree bvhC;
                bvhC.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
                auto visC = Reg3DCustom::extractVisibleVerticesCustom(
                    *liverMesh3D, bvhC, OrbitCam.cameraPos, OrbitCam.cameraTarget);
                if (visC.cloud->size() < 50) {
                    std::cerr << "[Key C] Not enough visible points" << std::endl;
                    break;
                }
                g_cluster1Points = visC.points;
                g_cluster2Points.clear();
                g_refineVertexIndices = visC.vertexIndices;
                computeIdealVoxelSizes();
                {
                    auto organs = getOrganList();
                    Reg3DCustom::performRegistrationSingleMesh(
                        organs, liverMesh3D, visC.vertexIndices,
                        screenMesh, OrbitCam.cameraPos,
                        gGridWidth, gGridHeight(), 15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);
                    computeUnifiedMetrics();
                    float rmse_before = registrationHandle.compRmse;
                    std::cout << "[Key C] HemiAuto compRMSE: " << rmse_before << std::endl;

                    CmaesRefine::Params p;
                    p.verbose   = true;
                    p.log_every = 30;
                    p.tx_range = 1.0f;
                    p.ty_range = 1.0f;
                    p.tz_range = 1.0f;
                    p.rx_range = 20.0f;
                    p.ry_range = 20.0f;
                    p.rz_range = 20.0f;
                    p.scale_lo = 0.85f;
                    p.scale_hi = 1.15f;
                    CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                                             gGridWidth, gGridHeight(), gDepthScale, p);
                    computeUnifiedMetrics();
                    std::cout << "[Key C] Result: " << rmse_before
                              << " -> " << registrationHandle.compRmse
                              << (r.improved ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;
                    poseSaveToLibrary();
                }
            }
        }
        if(currentMainMode == DEFORM_MODE){
            deformHandlPlace.reset();
            if (multiBody) {
                multiBody->fullReset();
            }
            std::cout << "Complete reset performed" << std::endl;

            multiBody->setRigidMode(true);
            multiBody->initPhysics();
            deformHandlPlace.state = DeformHandlPlaceData::HANDLE_PLACE_MODE;
            std::cout << "=== HandlePlace Mode Started (max "
                      << SoftBody::MAX_HANDLE_GROUPS << " groups) ===" << std::endl;
            std::cout << "Current groups: " << multiBody->handleGroups.size()
                      << "/" << SoftBody::MAX_HANDLE_GROUPS << std::endl;
        }
        break;

    case GLFW_KEY_A:
        if(currentMainMode == REGISTRATION_MODE){
            if (mode & GLFW_MOD_SHIFT) {
                saveARimageOrtho = true;
                std::cout << "[AR-Ortho] Shift+A pressed: ortho screenshot queued" << std::endl;
            } else {
                saveARimage = true;
            }
        } else if (currentMainMode == DEFORM_MODE && multiBody && liverMesh3D) {
            if (gAutoCtrl.numMove() == 0) {
                std::cout << "[KeyA] run handle gen (key 5) first" << std::endl;
                break;
            }
            gAutoCtrl.selectNextMove();
            deformHandlPlace.state = DeformHandlPlaceData::DEFORM_MODE;
        }
        break;

    case GLFW_KEY_F2:
        OrbitCam.resetToInitialState();
        std::cout << "Camera reset to initial position" << std::endl;
        break;

    case GLFW_KEY_S:
        if (currentMainMode == REGISTRATION_MODE) {
            poseSaveToLibrary();
        }
        break;

    case GLFW_KEY_Q:
        g_poseLibrary.showWindow = !g_poseLibrary.showWindow;
        std::cout << "[PoseLibrary] Window " << (g_poseLibrary.showWindow ? "ON" : "OFF") << std::endl;
        break;

    case GLFW_KEY_X:
        if (currentMainMode == REGISTRATION_MODE && g_poseLibrary.hasLastRegistration) {
            poseUndo();
        } else if (currentMainMode == DEFORM_MODE && multiBody) {
            gAutoDeformDebugBaseline = multiBody->getPositions();
            gAutoDeformDebugFrames = 60;
            std::cout << "[DBG X pressed] start 60-frame trace, baseline captured."
                      << " rigid=" << multiBody->isRigidMode()
                      << " att=" << multiBody->getNumAttachments()
                      << " sub=" << (int)deformHandlPlace.state << std::endl;
        }
        break;

    case GLFW_KEY_MINUS:
        if (currentMainMode == DEFORM_MODE) {
            gMoveScale = std::max(0.1f, gMoveScale - 0.1f);
            std::cout << "[MoveScale] = " << gMoveScale
                      << "  (handle will pull " << (gMoveScale * 100.0f) << "% of corresp distance)" << std::endl;
        }
        break;

    case GLFW_KEY_EQUAL:
        if (currentMainMode == DEFORM_MODE) {
            gMoveScale = std::min(2.0f, gMoveScale + 0.1f);
            std::cout << "[MoveScale] = " << gMoveScale
                      << "  (handle will pull " << (gMoveScale * 100.0f) << "% of corresp distance)" << std::endl;
        }
        break;

    case GLFW_KEY_BACKSPACE:
        if (currentMainMode == DEFORM_MODE && multiBody) {
            if (gAutoCtrl.activeMoveIdx() < 0) {
                std::cout << "[Bksp] no active move" << std::endl;
                break;
            }
            int activeIdx = gAutoCtrl.activeMoveIdx();
            float prevProg = gAutoCtrl.moveHandle(activeIdx).progress;
            if (prevProg <= 1e-5f) {
                std::cout << "[Bksp] move=" << activeIdx << " already at origin, no-op" << std::endl;
                break;
            }
            float prevRMSE = gAutoDeform.rmseLastMeasured;
            bool changed = gAutoCtrl.stepActive(multiBody, -gMoveScale);
            if (!changed) {
                std::cout << "[Bksp] no change" << std::endl;
                break;
            }
            gAutoCtrl.runBoost(multiBody, gAutoDeformBoostIter, gAutoDeformBoostDamping);

            float newProg = gAutoCtrl.moveHandle(activeIdx).progress;
            const auto& visVertsForRMSE = multiBody->getVisPositions(0);
            float immRMSE = AutoDeform::measureRMSE(
                gAutoDeform, visVertsForRMSE, screenMesh,
                gGridWidth, gGridHeight(), gDepthScale,
                /*storeAsBefore=*/false, 1.0f, /*verbose=*/false);
            float stepD = (prevRMSE > 0.0f) ? (immRMSE - prevRMSE) : 0.0f;
            float baseD = (gAutoDeform.rmseBeforeDeform > 0.0f) ? (immRMSE - gAutoDeform.rmseBeforeDeform) : 0.0f;
            std::cout << "[Bksp] move=" << activeIdx << " progress " << prevProg << " -> " << newProg
                      << (newProg <= 0.0f ? "  [AT ORIGIN]" : "")
                      << "  RMSE=" << immRMSE
                      << "  step:" << (stepD>=0?"+":"") << stepD << (stepD<0?" DOWN":" UP")
                      << "  base:" << (baseD>=0?"+":"") << baseD << (baseD<0?" DOWN":" UP")
                      << std::endl;

            gPendingRmseImmediate = immRMSE;
            gPendingRmseFrames    = 60;
        }
        break;

    case GLFW_KEY_TAB:
        if(currentMainMode == REGISTRATION_MODE){
            OrbitCam.switchTarget();
        }
        break;

    case GLFW_KEY_E:
        if(currentMainMode == REGISTRATION_MODE){
            bool shiftHeld = (mode & GLFW_MOD_SHIFT) != 0;
            bool ctrlHeld  = (mode & GLFW_MOD_CONTROL) != 0;

            /* =================================================================
               Ctrl+Shift+E = 軌道検索 (Shift+T 相当)
               Shift+V で得た 3D 寄り解 (P_V) と Shift+E で得た 2D 寄り解 (P_E) の
               間を SRT 空間で補間し、11 点で metrics を測定。
               論文方針: 単一の "best pose" を選択せず軌道全体を報告する。
               文献根拠: Alderliesten et al. SPIE 2013 が凹 Pareto front では
               どんな重み付けでも真に最適な解を見つけられないことを示しており、
               最近の MOREA (GECCO 2023) や Joutard et al. (SPIE 2022) も
               Pareto front 全体の報告を支持している。

               - 補間方式: Rotation=SLERP, Translation=linear, Scale=log-linear
               - 出力: depth_output/traj_<timestamp>/
                 * pareto.csv: 全 11 点の metrics
                 * t_XX.png: 各 t の overlay スクリーンショット (肝臓+全 organ+texture)
                 * t_XX_seg8.png: 各 t のセグメント 8 bump mask (黒背景に白、shrunken-liver で occlude)
                 * t_XX_image.png: 各 t の screenMesh のみ (物理画像、mask との直接比較用)
               - 最終姿勢: t=0.5 (幾何的中点、単一 best 選択なし)
               - Pose Library には保存しない (実験モード)
               - 事前条件: Shift+V と Shift+E を同セッション内で両方実行済み
               ================================================================= */
            if (shiftHeld && ctrlHeld) {
                const std::string keyLabel = "Ctrl+Shift+E";
                std::cout << "\n=== Trajectory Search (" << keyLabel
                          << ", SRT interpolation between Shift+V and Shift+E poses) ===" << std::endl;

                if (!gShiftV_lastValid || !gShiftE_lastValid) {
                    std::cerr << "[" << keyLabel << "] Both Shift+V and Shift+E must be "
                              << "executed first in this session.  Status: "
                              << "ShiftV=" << (gShiftV_lastValid ? "OK" : "MISSING")
                              << ", ShiftE=" << (gShiftE_lastValid ? "OK" : "MISSING") << std::endl;
                    break;
                }
                if (g_initOrganVertices.empty() || !liverMesh3D) {
                    std::cerr << "[" << keyLabel << "] Initial pose snapshot missing." << std::endl;
                    break;
                }

                /* 端点姿勢を R/T/S に分解。T = [s*R | t; 0 0 0 1] の形なので、
                   左上 3x3 から scale = cbrt(det), R = M/scale を抽出。
                   glm は列優先 (T[col][row])。 */
                auto decomposeSRT = [](const glm::mat4& T,
                                       glm::quat& outQ,
                                       glm::vec3& outT,
                                       float&     outS) {
                    glm::mat3 M(T);
                    float det = glm::determinant(M);
                    /* 負の determinant はミラー反転。Procrustes 後なら通常は正。 */
                    float s = std::cbrt(std::max(det, 1e-8f));
                    glm::mat3 R = M / s;
                    /* 数値的な再直交化: R^T R = I に寄せる */
                    outQ = glm::normalize(glm::quat_cast(R));
                    outT = glm::vec3(T[3]);
                    outS = s;
                };

                glm::quat qV, qE;
                glm::vec3 tV, tE;
                float     sV, sE;
                decomposeSRT(gShiftV_lastTransform, qV, tV, sV);
                decomposeSRT(gShiftE_lastTransform, qE, tE, sE);

                /* quat 符号合わせ (short path) */
                if (glm::dot(qV, qE) < 0.0f) qE = -qE;

                std::cout << "[" << keyLabel << "] Endpoint V: scale=" << sV
                          << " translate=(" << tV.x << "," << tV.y << "," << tV.z << ")"
                          << std::endl;
                std::cout << "[" << keyLabel << "] Endpoint E: scale=" << sE
                          << " translate=(" << tE.x << "," << tE.y << "," << tE.z << ")"
                          << std::endl;

                /* 現在姿勢を初期値として使う (f_norm の正規化定数) */
                auto organs = getOrganList();
                computeUnifiedMetrics();
                float compRmse_init = registrationHandle.compRmse;
                /* 2D Hausdorff と IoU を得るため Sil2D-Fast を直接呼ぶ */
                int dI0, dU0; double dMs0; float dH0;
                bool wasQuiet0 = g_quietMetrics;
                g_quietMetrics = true;
                CmaesRefine::computeSilhouette2DObjectiveFast(
                    liverMesh3D, view, projection, 8, &dI0, &dU0, &dMs0, &dH0);
                g_quietMetrics = wasQuiet0;
                float iou_init = (dU0 > 0) ? (float)dI0 / (float)dU0 : 0.0f;

                /* 正規化に使う分母。0 除算回避 */
                float denom_iou  = std::max(1.0f - iou_init,      1e-6f);
                float denom_rmse = std::max(compRmse_init,        1e-6f);

                std::cout << "[" << keyLabel << "] Init: compRMSE=" << compRmse_init
                          << "  IoU=" << iou_init
                          << "  (normalization anchors)" << std::endl;

                /* 軌道データ用タイムスタンプ付きサブディレクトリを事前作成。
                   スクリーンショット と CSV を同じフォルダに格納。 */
                std::string trajDir;
                std::string tsbufStr;
                {
                    std::time_t tnow = std::time(nullptr);
                    std::tm tm;
#ifdef _WIN32
                    localtime_s(&tm, &tnow);
#else
                    localtime_r(&tnow, &tm);
#endif
                    char tsbuf[32];
                    std::strftime(tsbuf, sizeof(tsbuf), "%Y%m%d_%H%M%S", &tm);
                    tsbufStr = tsbuf;
                    trajDir = std::string(DEPTH_OUTPUT_PATH) + "traj_" + tsbufStr;
                    try {
                        std::filesystem::create_directories(trajDir);
                    } catch (const std::exception& e) {
                        std::cerr << "[" << keyLabel << "] Failed to create traj dir: "
                                  << e.what() << std::endl;
                    }
                    std::cout << "[" << keyLabel << "] Trajectory output dir: "
                              << trajDir << std::endl;
                }

                /* trajectory 全体で不変な物理画像を 1 枚だけ保存。
                   depth slider を 0 に一時的に下げて screen mesh を平面化してから撮る。
                   (depth が非 0 だと screenMesh が深度方向に歪むため) */
                saveTrajectoryImageOnce(trajDir, view, projection, model, OrbitCam.cameraPos);

                /* 11 サンプルで走査 */
                const int N_SAMPLES = 11;
                std::vector<float> tvals(N_SAMPLES), rmseArr(N_SAMPLES), iouArr(N_SAMPLES);
                std::vector<float> avgArr(N_SAMPLES), maxT2SArr(N_SAMPLES), maxS2TArr(N_SAMPLES);
                std::vector<float> hdArr(N_SAMPLES), h2dArr(N_SAMPLES), fnormArr(N_SAMPLES);

                for (int k = 0; k < N_SAMPLES; k++) {
                    float t = (float)k / (float)(N_SAMPLES - 1);
                    tvals[k] = t;

                    /* 補間: quat SLERP, translation linear, scale log-linear */
                    glm::quat qt = glm::slerp(qV, qE, t);
                    glm::vec3 tt = (1.0f - t) * tV + t * tE;
                    float     st = std::exp((1.0f - t) * std::log(std::max(sV, 1e-8f))
                                           + t * std::log(std::max(sE, 1e-8f)));

                    glm::mat3 R = glm::mat3_cast(qt);
                    glm::mat4 Tk(1.0f);
                    Tk[0] = glm::vec4(st * R[0], 0.0f);
                    Tk[1] = glm::vec4(st * R[1], 0.0f);
                    Tk[2] = glm::vec4(st * R[2], 0.0f);
                    Tk[3] = glm::vec4(tt, 1.0f);

                    /* 姿勢適用 (初期メッシュから) */
                    PoseLibrary::applyTransformToMeshes(
                        Tk, g_initOrganVertices, g_initOrganNormals, organs);
                    for (auto* m : organs) if (m) setUp(*m);

                    /* 指標測定 */
                    computeUnifiedMetrics();
                    float rmse_k   = registrationHandle.compRmse;
                    float avg_k    = registrationHandle.compAvgError;
                    float mT2S_k   = registrationHandle.compMaxError;
                    float mS2T_k   = registrationHandle.compMaxS2T;
                    float hd_k     = registrationHandle.compHausdorff;
                    float h2d_k    = registrationHandle.sil2DHausdorff;

                    /* IoU を Fast 版で再測 (computeUnifiedMetrics 内では Hausdorff2D
                       しか保存していないため。実質はダブルカウントだが安全のため) */
                    int dIk, dUk; double dMsk; float dHk;
                    bool wq = g_quietMetrics;
                    g_quietMetrics = true;
                    CmaesRefine::computeSilhouette2DObjectiveFast(
                        liverMesh3D, view, projection, 8, &dIk, &dUk, &dMsk, &dHk);
                    g_quietMetrics = wq;
                    float iou_k = (dUk > 0) ? (float)dIk / (float)dUk : 0.0f;

                    /* f_norm は CSV に記録するが姿勢選択には使わない (初期値依存で不安定)。
                       論文的には軌道全体を報告し単一の「best」は選ばない方針
                       (Alderliesten 2013, Joutard 2022 等の Pareto 文献に整合)。 */
                    float fnorm = (1.0f - iou_k) / denom_iou + rmse_k / denom_rmse;

                    rmseArr[k]   = rmse_k;
                    iouArr[k]    = iou_k;
                    avgArr[k]    = avg_k;
                    maxT2SArr[k] = mT2S_k;
                    maxS2TArr[k] = mS2T_k;
                    hdArr[k]     = hd_k;
                    h2dArr[k]    = h2d_k;
                    fnormArr[k]  = fnorm;

                    std::cout << "[" << keyLabel << "] t=" << std::fixed << std::setprecision(2) << t
                              << "  compRMSE=" << std::setprecision(4) << rmse_k
                              << "  IoU=" << iou_k
                              << "  H3D=" << hd_k
                              << "  H2D=" << std::setprecision(1) << h2d_k << "px"
                              << std::endl;
                    std::cout << std::defaultfloat << std::setprecision(6);

                    /* スクリーンショット保存 (overlay + seg8-only)。
                       Python 側で seg8 の annotation と比較する用。 */
                    saveTrajectoryFrame(trajDir, t, view, projection, model,
                                        OrbitCam.cameraPos);
                }

                /* CSV 出力 (軌道ディレクトリ内に配置) */
                {
                    std::string csvPath = trajDir + "/pareto.csv";
                    std::ofstream ofs(csvPath);
                    if (ofs.is_open()) {
                        ofs << "t,compRMSE,compAvgError,compMaxT2S,compMaxS2T,"
                            << "compHausdorff3D,IoU,sil2DHausdorff,f_norm\n";
                        ofs << std::fixed << std::setprecision(6);
                        for (int k = 0; k < N_SAMPLES; k++) {
                            ofs << tvals[k] << "," << rmseArr[k] << "," << avgArr[k]
                                << "," << maxT2SArr[k] << "," << maxS2TArr[k]
                                << "," << hdArr[k] << "," << iouArr[k]
                                << "," << h2dArr[k] << "," << fnormArr[k] << "\n";
                        }
                        ofs.close();
                        std::cout << "[" << keyLabel << "] CSV saved: " << csvPath << std::endl;
                    } else {
                        std::cerr << "[" << keyLabel << "] Failed to open CSV: " << csvPath
                                  << std::endl;
                    }
                }

                /* 最終姿勢として t=0.5 (幾何的中点) を適用。
                   論文 scope: 単一の best を選択せず軌道全体を報告するのが主貢献。
                   UI 上は何らかの姿勢を適用する必要があるため、偏りのない中点を
                   デフォルトとする。ユーザは Pose Library から Shift+V 姿勢 (t=0)
                   や Shift+E 姿勢 (t=1) に戻せる。 */
                {
                    const float t_default = 0.5f;
                    int default_idx = N_SAMPLES / 2;  /* = 5 for N_SAMPLES=11 */
                    glm::quat qt = glm::slerp(qV, qE, t_default);
                    glm::vec3 tt = (1.0f - t_default) * tV + t_default * tE;
                    float     st = std::exp((1.0f - t_default) * std::log(std::max(sV, 1e-8f))
                                           + t_default * std::log(std::max(sE, 1e-8f)));
                    glm::mat3 R = glm::mat3_cast(qt);
                    glm::mat4 Tk(1.0f);
                    Tk[0] = glm::vec4(st * R[0], 0.0f);
                    Tk[1] = glm::vec4(st * R[1], 0.0f);
                    Tk[2] = glm::vec4(st * R[2], 0.0f);
                    Tk[3] = glm::vec4(tt, 1.0f);
                    PoseLibrary::applyTransformToMeshes(
                        Tk, g_initOrganVertices, g_initOrganNormals, organs);
                    for (auto* m : organs) if (m) setUp(*m);
                    computeUnifiedMetrics();
                    std::cout << "[" << keyLabel << "] Default pose applied: t=" << t_default
                              << "  (midpoint, no 'best' selection)"
                              << "  compRMSE=" << rmseArr[default_idx]
                              << "  IoU=" << iouArr[default_idx]
                              << "  H3D=" << hdArr[default_idx]
                              << "  H2D=" << h2dArr[default_idx] << "px"
                              << std::endl;
                    std::cout << "[" << keyLabel << "] Experimental mode: "
                              << "NOT saved to pose library. Full trajectory in CSV; "
                              << "screenshots (overlay + seg8-only) saved per t-sample."
                              << std::endl;
                }
                break;
            }

            /* Shift+E = 2D IoU BIPOP-CMA-ES (高速ラスタライズ版)
               実装は runShiftE() を参照。UI ボタン "Silhouette Alignment" と共通。 */
            if (shiftHeld) {
                runShiftE();
                break;
            }

            /* Key E (Shift なし) = 従来の HemiAuto + Boundary-Weighted CMA-ES。
               まだゼロから自動登録をしたいケース用に残す。 */
            std::cout << "\n=== HemiAuto + Boundary-Weighted CMA-ES (Key E) ===" << std::endl;
            gUIManager.state.regMethod = 1;
            poseAutoSaveBeforeRegistration();
            resetRegistrationState();

            Reg3D::BVHTree bvhE;
            bvhE.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
            auto visE = Reg3DCustom::extractVisibleVerticesCustom(
                *liverMesh3D, bvhE, OrbitCam.cameraPos, OrbitCam.cameraTarget);
            if (visE.cloud->size() < 50) {
                std::cerr << "[Key E] Not enough visible points" << std::endl;
                break;
            }
            g_cluster1Points = visE.points;
            g_cluster2Points.clear();
            g_refineVertexIndices = visE.vertexIndices;
            computeIdealVoxelSizes();
            {
                auto organs = getOrganList();
                Reg3DCustom::performRegistrationSingleMesh(
                    organs, liverMesh3D, visE.vertexIndices,
                    screenMesh, OrbitCam.cameraPos,
                    gGridWidth, gGridHeight(), 15, 0.005f, 0.35f, true, 0.03f, gDepthScale, g_voxelSize);
                computeUnifiedMetrics();
                float rmse_before = registrationHandle.compRmse;
                std::cout << "[Key E] HemiAuto compRMSE: " << rmse_before << std::endl;

                CmaesRefine::Params p;
                p.verbose     = true;
                p.log_every   = 30;
                p.tx_range = 1.0f;
                p.ty_range = 1.0f;
                p.tz_range = 1.0f;
                p.rx_range = 20.0f;
                p.ry_range = 20.0f;
                p.rz_range = 20.0f;
                p.scale_lo = 0.85f;
                p.scale_hi = 1.15f;
                p.use_boundary_weight = true;
                p.boundary_width      = 12.0f;
                p.boundary_boost      = 3.0f;

                CmaesRefine::Result r = CmaesRefine::run(organs, screenMesh,
                                                         gGridWidth, gGridHeight(), gDepthScale, p);
                computeUnifiedMetrics();
                std::cout << "[Key E] Result: " << rmse_before
                          << " -> " << registrationHandle.compRmse
                          << (r.improved ? " [IMPROVED]" : " [NO CHANGE]") << std::endl;
                poseSaveToLibrary();
            }
        }
        break;

    }
}


void glfw_OnFramebufferSize(GLFWwindow* window, int width, int height)
{
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, gWindowWidth, gWindowHeight);

    g_sceneTexAllocated = false;

    OrbitCam.onWindowResize(width, height);
}

void showFPS(GLFWwindow* window)
{
    static double previousSeconds = 0.0;
    static int frameCount = 0;
    double elapsedSeconds;
    double currentSeconds = glfwGetTime();

    elapsedSeconds = currentSeconds - previousSeconds;

    if (elapsedSeconds > 0.25)
    {
        previousSeconds = currentSeconds;
        double fps = (double)frameCount / elapsedSeconds;
        double msPerFrame = 1000.0 / fps;

        int gcd = std::gcd(gWindowWidth, gWindowHeight);
        int aspectWidth = gWindowWidth / gcd;
        int aspectHeight = gWindowHeight / gcd;

        double aspectRatio = (double)gWindowWidth / (double)gWindowHeight;

        std::ostringstream outs;
        outs.precision(3);
        outs << std::fixed
             << "FPS: " << fps << "    "
             << "Frame Time: " << msPerFrame << " (ms)    "
             << "Window: " << gWindowWidth << "x" << gWindowHeight << "    "
             << "Aspect: " << aspectWidth << ":" << aspectHeight
             << " (" << aspectRatio << ")";
        glfwSetWindowTitle(window, outs.str().c_str());

        frameCount = 0;
    }

    frameCount++;
}
