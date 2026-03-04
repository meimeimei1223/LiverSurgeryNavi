#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <ctime>
#include <functional>

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

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "RegistrationImGuiManager.h"
#include "PoseLibrary.h"

CameraPreview gCameraPreview;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DepthRunner gDepthRunner;

std::vector<DepthRunnerPoint> gUserSegPoints;
std::vector<glm::vec3>        gUserSegPoints3D;
std::vector<bool>             gUserSegPointsFG;

float gDepthScale = 0.3f;
const float gMeshScale = 10.0f;
int   gGridWidth = 128;

int gWindowWidth = 1280, gWindowHeight = 720;
GLFWwindow* gWindow = NULL;

std::string gDroppedFilePath = "";
bool        gFileDropped     = false;

RegistrationImGuiManager gUIManager;

glm::vec3 hit_position;
int hit_index;
bool isDragging;

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

            physicsObject->smartGrab(hit.position, 0.5f);
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

MainMode currentMainMode = REGISTRATION_MODE;

bool saveARimage = false;
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

static void poseAutoSaveBeforeRegistration() {
    auto organs = getOrganList();
    PoseEntry backup;
    backup.timestamp = PoseLibrary::nowTimestamp();
    PoseLibrary::snapshotMeshes(backup, organs);
    g_poseLibrary.autoSaveLastRegistration(backup);
}

// -------------------------------------------------------
// Unified comparison metric computation
// Same method for ALL entries: Liver visible vertices → depth front face
// Visible vertex INDICES are LOCKED at first call after registration.
// On subsequent calls (refine, manual save), the same indices are reused
// but current vertex POSITIONS are read — so the metric reflects the
// actual mesh geometry while keeping the source population identical.
// -------------------------------------------------------
static void computeUnifiedMetrics() {
    Reg3DCustom::NoOpen3DRegistration reg;
    auto targetCloud = reg.extractFrontFacePoints(*screenMesh, 128, 72, gDepthScale);

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

    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "[UnifiedMetrics T->S] Target: " << targetCloud->size()
              << "  Matched: " << registrationHandle.compCount
              << "  RMSE: " << registrationHandle.compRmse
              << "  AvgErr: " << registrationHandle.compAvgError
              << "  MaxErr: " << registrationHandle.compMaxError << std::endl;
}

static void poseSaveToLibrary() {
    if (registrationHandle.state != RegistrationData::REGISTERED) {
        std::cout << "[PoseLibrary] No registration to save" << std::endl;
        return;
    }
    auto organs = getOrganList();
    PoseEntry::Method method;
    if (gUIManager.state.regMethod == 0) method = PoseEntry::FULL_AUTO;
    else if (gUIManager.state.regMethod == 1) method = PoseEntry::HEMI_AUTO;
    else method = PoseEntry::UMEYAMA;

    g_poseLibrary.saveCurrentToLibrary(
        method,
        registrationHandle.refineCount,
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
        registrationHandle.compCount,
        registrationHandle.compSource,
        registrationHandle.compTarget,
        organs);
}

static void poseApplyEntry(int entryId) {
    auto organs = getOrganList();
    if (g_poseLibrary.applyEntry(entryId, organs)) {
        registrationHandle.state = RegistrationData::REGISTERED;
        registrationHandle.useRegistration = true;
        for (auto& e : g_poseLibrary.entries) {
            if (e.id == entryId) {
                registrationHandle.fitness       = e.baseFitness;
                registrationHandle.icpRmse       = e.baseIcpRmse;
                registrationHandle.averageError  = e.baseAvgError;
                registrationHandle.rmse          = e.baseRmse;
                registrationHandle.maxError      = e.baseMaxError;
                registrationHandle.scaleFactor   = e.baseScale;
                registrationHandle.refineCount   = e.refineCount;
                registrationHandle.refineInitialRMSE   = e.refineInitialRMSE;
                registrationHandle.refineBestRMSE      = e.refineBestRMSE;
                registrationHandle.refineBestIteration = e.refineBestIteration;
                registrationHandle.compRmse     = e.compRmse;
                registrationHandle.compAvgError = e.compAvgError;
                registrationHandle.compMaxError = e.compMaxError;
                registrationHandle.compCount    = e.compCount;
                registrationHandle.compSource   = e.corrSource;
                registrationHandle.compTarget   = e.corrTarget;
                if (e.baseMethod == PoseEntry::FULL_AUTO)
                    gUIManager.state.regMethod = 0;
                else if (e.baseMethod == PoseEntry::HEMI_AUTO)
                    gUIManager.state.regMethod = 1;
                else
                    gUIManager.state.regMethod = 2;
                break;
            }
        }
    }
}

static void poseUndo() {
    auto organs = getOrganList();
    g_poseLibrary.undoToLast(organs);
    registrationHandle.state = RegistrationData::REGISTERED;
    registrationHandle.useRegistration = true;
    computeUnifiedMetrics();
}

// -------------------------------------------------------
// Pose Library ImGui Window
// -------------------------------------------------------
static void drawPoseLibraryWindow() {
    if (!g_poseLibrary.showWindow) return;

    ImGui::SetNextWindowSize(ImVec2(560, 400), ImGuiCond_FirstUseEver);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.08f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.12f, 0.10f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.20f, 0.15f, 0.30f, 1.0f));

    if (ImGui::Begin("Pose Library", &g_poseLibrary.showWindow)) {
        ImGui::Text("Entries: %d / %d", (int)g_poseLibrary.entries.size(), g_poseLibrary.maxEntries);
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
        if (ImGui::Button("Export CSV", ImVec2(120, 0))) {
            g_poseLibrary.exportToCsv("pose_library_export.csv");
        }
        ImGui::Separator();

        // Header
        ImGui::Columns(8, "pose_cols", true);
        ImGui::SetColumnWidth(0, 26);
        ImGui::SetColumnWidth(1, 80);
        ImGui::SetColumnWidth(2, 36);
        ImGui::SetColumnWidth(3, 80);
        ImGui::SetColumnWidth(4, 55);
        ImGui::SetColumnWidth(5, 50);
        ImGui::SetColumnWidth(6, 42);
        ImGui::SetColumnWidth(7, 34);
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "#");         ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "Method");    ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "Ref");       ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "Comp RMSE"); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "N");         ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "");          ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "");          ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "");          ImGui::NextColumn();
        ImGui::Separator();

        int deleteId = -1;
        int applyId  = -1;
        int exportCorrId = -1;

        for (size_t i = 0; i < g_poseLibrary.entries.size(); i++) {
            auto& e = g_poseLibrary.entries[i];
            bool isActive = (e.id == g_poseLibrary.activeEntryId);

            if (isActive) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));
            }

            ImGui::Text("%d", (int)(i + 1)); ImGui::NextColumn();

            // Method (short name) + tooltip for details
            ImGui::Text("%s", e.methodStr());
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("ID: %d", e.id);
                ImGui::Text("Time: %s", e.timestamp.c_str());
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f,1.0f,0.5f,1), "=== Unified (Liver visible vs Depth) ===");
                ImGui::Text("Comp RMSE:  %.6f  (%d pairs)", e.compRmse, e.compCount);
                ImGui::Text("Comp AvgErr: %.6f", e.compAvgError);
                ImGui::Text("Comp MaxErr: %.6f", e.compMaxError);
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "--- Base Registration ---");
                ImGui::Text("Fitness (ICP): %.6f", e.baseFitness);
                ImGui::Text("ICP RMSE: %.6f", e.baseIcpRmse);
                ImGui::Text("Corr. RMSE (sampled): %.6f", e.baseRmse);
                ImGui::Text("Corr. AvgErr: %.6f", e.baseAvgError);
                ImGui::Text("Corr. MaxErr: %.6f", e.baseMaxError);
                ImGui::Text("Scale: %.4f", e.baseScale);
                if (e.refineCount > 0) {
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "--- Refine ---");
                    ImGui::Text("Refine Count: %d", e.refineCount);
                    ImGui::Text("Refine Init RMSE (full): %.6f", e.refineInitialRMSE);
                    ImGui::Text("Refine Best RMSE (full): %.6f", e.refineBestRMSE);
                    ImGui::Text("Refine Best Iter: %d", e.refineBestIteration);
                }
                ImGui::EndTooltip();
            }
            ImGui::NextColumn();

            // Refine column
            if (e.refineCount > 0) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "x%d", e.refineCount);
            } else {
                ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "-");
            }
            ImGui::NextColumn();

            ImGui::Text("%.4f", e.compRmse); ImGui::NextColumn();
            ImGui::Text("%d", e.compCount);   ImGui::NextColumn();

            ImGui::PushID(e.id);
            if (ImGui::SmallButton("Apply")) {
                applyId = e.id;
            }
            ImGui::NextColumn();
            if (ImGui::SmallButton("Corr")) {
                exportCorrId = e.id;
            }
            ImGui::NextColumn();
            if (ImGui::SmallButton("Del")) {
                deleteId = e.id;
            }
            ImGui::PopID();
            ImGui::NextColumn();

            if (isActive) {
                ImGui::PopStyleColor();
            }
        }

        ImGui::Columns(1);

        if (applyId >= 0) {
            poseApplyEntry(applyId);
        }
        if (deleteId >= 0) {
            g_poseLibrary.deleteEntry(deleteId);
        }
        if (exportCorrId >= 0) {
            std::string fname = "pose_" + std::to_string(exportCorrId) + "_correspondences.csv";
            g_poseLibrary.exportCorrespondences(exportCorrId, fname);
        }

        ImGui::Separator();
        ImGui::Spacing();

        // Bottom buttons
        float bw = (ImGui::GetContentRegionAvail().x - 8) / 2.0f;
        bool canUndo = g_poseLibrary.hasLastRegistration;

        if (!canUndo) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.4f);
        if (ImGui::Button("Undo", ImVec2(bw, 28))) {
            if (canUndo) poseUndo();
        }
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
        gDepthScale = 0.3f;
        clearSegPoints();
        depthSplitScreenMode = false;
        splitScreenMode = false;
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
        regenerateDepthMesh(screenMesh, gDepthScale, gMeshScale);
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
        std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                          tumorMesh3D, segmentMesh3D, gbMesh3D};
        std::vector<std::string> names = {"Liver","Portal","Vein","Tumor","Segment","Gallbladder"};
        Reg3DCustom::performRegistrationMultiMeshWithScale(
            organs, names, screenMesh, OrbitCam.cameraPos,
            128, 72, 15, 0.005f, 0.35f, true, 0.03f, gDepthScale);
        computeUnifiedMetrics();
        poseSaveToLibrary();
    };

    a.onHemiAuto = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        gUIManager.state.regMethod = 1;
        poseAutoSaveBeforeRegistration();
        resetRegistrationState();
        static Reg3D::BVHTree bvh;
        bvh.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
        auto vis = Reg3DCustom::extractVisibleVerticesCustom(
            *liverMesh3D, bvh, OrbitCam.cameraPos, OrbitCam.cameraTarget);
        if (vis.cloud->size() < 50) return;
        g_cluster1Points = vis.points;
        g_cluster2Points.clear();
        g_refineVertexIndices = vis.vertexIndices;
        std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                          tumorMesh3D, segmentMesh3D, gbMesh3D};
        Reg3DCustom::performRegistrationSingleMesh(
            organs, liverMesh3D, vis.vertexIndices,
            screenMesh, OrbitCam.cameraPos,
            128, 72, 15, 0.005f, 0.35f, true, 0.03f, gDepthScale);
        computeUnifiedMetrics();
        poseSaveToLibrary();
    };

    a.onRefine = []() {
        if (currentMainMode != REGISTRATION_MODE) return;
        if (registrationHandle.state == RegistrationData::REGISTERED) {
            if (g_refineVertexIndices.empty()) {
                std::cerr << "[Refine] No visible vertex indices." << std::endl;
                return;
            }
            // Save before refine starts
            poseAutoSaveBeforeRegistration();
            std::cout << "\n=== Normal-Compatible Refinement START ===" << std::endl;
            std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D,
                                              tumorMesh3D, segmentMesh3D, gbMesh3D};
            NormalRefine::RefineParams params;
            if (NormalRefine::initRefine(g_refineState, liverMesh3D,
                                         g_refineVertexIndices,
                                         screenMesh, organs,
                                         128, 72, gDepthScale, params,
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
        registrationHandle.resetTransformOnly();
        splitScreenMode = false;
        gUIManager.state.regMethod = -1;
        g_refineVertexIndices.clear();
        g_refineState.reset();
    };

    a.onPoseUndo = []() {
        poseUndo();
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
        reloadMesh(liverMesh3D,   TARGET_FILE_PATH,  glm::vec3(0.8f, 0.2f, 0.2f));
        reloadMesh(portalMesh3D,  PORTAL_FILE_PATH,  glm::vec3(0.2f, 0.2f, 0.8f));
        reloadMesh(veinMesh3D,    VEIN_FILE_PATH,    glm::vec3(0.2f, 0.5f, 0.5f));
        reloadMesh(tumorMesh3D,   TUMOR_FILE_PATH,   glm::vec3(0.8f, 0.5f, 0.5f));
        reloadMesh(segmentMesh3D, SEGMENT_FILE_PATH, glm::vec3(0.2f, 0.8f, 0.5f));
        reloadMesh(gbMesh3D,      GB_FILE_PATH,      glm::vec3(0.2f, 0.8f, 0.2f));

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

    for (int i = 0; i < 6; i++)
        s.organs[i].alpha = meshAlphaValues[i];

    s.boardAlpha = meshAlphaValues[6];

    s.clusterVis = g_showClusterVisualization;
    s.correspondenceVis = g_showCorrespondencePoints;
}

int main()
{
    initPaths();
    initFilePaths();
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

    OrbitCam.setIntrinsics(800.0f, 800.0f, gWindowWidth/2.0f, gWindowHeight/2.0f);

    OrbitCam.printCameraInfo();

    deformSphereMarker.generate(1.0f, 16, 16);
    deformSphereMarker.setup();

    liverMesh3D = new mCutMesh(liverMesh3D->loadMeshFromFile(TARGET_FILE_PATH.c_str()));
    liverMesh3D->mColor = glm::vec3(0.8f, 0.2f, 0.2f);
    setUp(*liverMesh3D);
    portalMesh3D = new mCutMesh(portalMesh3D->loadMeshFromFile(PORTAL_FILE_PATH.c_str()));
    portalMesh3D->mColor = glm::vec3(0.2f, 0.2f, 0.8f);
    setUp(*portalMesh3D);
    veinMesh3D = new mCutMesh(veinMesh3D->loadMeshFromFile(VEIN_FILE_PATH.c_str()));
    veinMesh3D->mColor = glm::vec3(0.2f, 0.5f, 0.5f);
    setUp(*veinMesh3D);
    tumorMesh3D = new mCutMesh(tumorMesh3D->loadMeshFromFile(TUMOR_FILE_PATH.c_str()));
    tumorMesh3D->mColor = glm::vec3(0.8f, 0.5f, 0.5f);
    setUp(*tumorMesh3D);
    segmentMesh3D = new mCutMesh(segmentMesh3D->loadMeshFromFile(SEGMENT_FILE_PATH.c_str()));
    segmentMesh3D->mColor = glm::vec3(0.2f, 0.8f, 0.5f);
    setUp(*segmentMesh3D);
    gbMesh3D = new mCutMesh(gbMesh3D->loadMeshFromFile(GB_FILE_PATH.c_str()));
    gbMesh3D->mColor = glm::vec3(0.2f, 0.8f, 0.2f);
    setUp(*gbMesh3D);

    allMeshes.push_back(liverMesh3D);
    allMeshes.push_back(portalMesh3D);
    allMeshes.push_back(veinMesh3D);
    allMeshes.push_back(tumorMesh3D);
    allMeshes.push_back(segmentMesh3D);
    allMeshes.push_back(gbMesh3D);

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
                    std::cout << "Drawing clusters: " << g_cluster1Points.size()
                    << " + " << g_cluster2Points.size()
                    << " + " << g_targetPoints.size() << std::endl;

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

                VoxelTetrahedralizer tetrahedralizer(DEFAULT_GRID_SIZE, Reg_TARGET_FILE_PATH, OUTPUT_TET_FILE);

                tetrahedralizer.setSmoothingEnabled(true, SMOOTH_ITERATION, SMOOTH_FACTOR, true);

                VoxelTetrahedralizer::InflationSettings inflationSettings;
                if(DEFAULT_GRID_SIZE<30){
                    inflationSettings.enabled = true;} else {inflationSettings.enabled = false;};
                inflationSettings.targetCoverage = 99.0f;
                inflationSettings.successThreshold = 99.0f;
                tetrahedralizer.setInflationSettings(inflationSettings);

                MeshDataTypes::SimpleMeshData resultData = tetrahedralizer.execute();

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
                    auto positions = multiBody->getHandleGroupPositions(g);
                    glm::vec3 color = getPointColor(g, true);

                    for (const auto& pos : positions) {
                        glm::vec3 worldPos = glm::vec3(model * glm::vec4(pos, 1.0f));
                        deformSphereMarker.draw(shaderProgram, worldPos, color, 0.2f,
                                                view, projection, OrbitCam.cameraPos);
                    }
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

        glViewport(0, 0, gWindowWidth, gWindowHeight);
        gUIManager.draw(gWindowWidth, gWindowHeight);

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

void glfw_onMouseScroll(GLFWwindow* window, double deltaX, double deltaY) {
    if (ImGui::GetIO().WantCaptureMouse) return;

    FullSphereCamera* activeCamera = getActiveCamera(window);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != 1) {
        activeCamera->gRadius += deltaY * activeCamera->ZOOM_SENSITIVITY;
        activeCamera->gRadius = glm::clamp(activeCamera->gRadius, 2.0f, 80.0f);
        std::cout << "Camera radius: " << activeCamera->gRadius << std::endl;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == 1) {
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

                    gGrabber->placeSphere(xpos, ypos, 1.0f);

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

    case GLFW_KEY_1:
    case GLFW_KEY_2:
    case GLFW_KEY_3:
    case GLFW_KEY_4:
    case GLFW_KEY_5:
    case GLFW_KEY_6:
    case GLFW_KEY_7:
    {
        int meshIndex = key - GLFW_KEY_1;

        if (meshIndex >= 0 && meshIndex < meshAlphaValues.size()) {
            float currentAlpha = meshAlphaValues[meshIndex];

            if (currentAlpha < 0.01f) {
                meshAlphaValues[meshIndex] = 0.5f;
            } else if (currentAlpha < 0.75f) {
                meshAlphaValues[meshIndex] = 1.0f;
            } else {
                meshAlphaValues[meshIndex] = 0.0f;
            }

            const char* meshNames[] = {"Liver", "Portal", "Vein", "Tumor", "Segment", "GB", "ScreenMesh"};
            std::cout << meshNames[meshIndex] << " alpha: "
                      << meshAlphaValues[meshIndex] << std::endl;
        }
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
            gDepthScale += 0.05f;
            regenerateDepthMesh(screenMesh, gDepthScale, gMeshScale);
        }
        break;

    case GLFW_KEY_DOWN:
        if (currentMainMode == REGISTRATION_MODE) {
            gDepthScale -= 0.05f;
            if (gDepthScale < 0.0f) gDepthScale = 0.0f;
            regenerateDepthMesh(screenMesh, gDepthScale, gMeshScale);
        }
        break;

    case GLFW_KEY_COMMA:
        if (currentMainMode == REGISTRATION_MODE) {
            int prevGW = gGridWidth;
            gGridWidth = std::max(64, gGridWidth / 2);
            if (gGridWidth != prevGW) {
                regenerateDepthMesh(screenMesh, gDepthScale, gMeshScale);
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
                regenerateDepthMesh(screenMesh, gDepthScale, gMeshScale);
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
            gDepthScale = 0.3f;
            clearSegPoints();
            depthSplitScreenMode = false;
            splitScreenMode = false;
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
            gDepthScale = 0.3f;
            clearSegPoints();
            depthSplitScreenMode = false;
            splitScreenMode = false;
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
                128, 72,
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
        if(currentMainMode == REGISTRATION_MODE) {
            std::cout << "\n============================================" << std::endl;
            std::cout << "  Camera View-Based Registration (Custom)" << std::endl;
            std::cout << "============================================\n" << std::endl;

            resetRegistrationState();

            std::cout << "Step 1: Building BVH..." << std::endl;
            static Reg3D::BVHTree cameraBvhTree;
            cameraBvhTree.build(liverMesh3D->mVertices, liverMesh3D->mIndices);
            std::cout << "  BVH nodes: " << cameraBvhTree.nodes.size() << std::endl;

            std::cout << "\nStep 2: Extracting visible vertices from camera view..." << std::endl;

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

            std::cout << "\nStep 3: Starting iterative registration..." << std::endl;

            std::vector<mCutMesh*> organs = {liverMesh3D, portalMesh3D, veinMesh3D, tumorMesh3D, segmentMesh3D, gbMesh3D};

            Reg3DCustom::performRegistrationSingleMesh(
                organs, liverMesh3D, visibility.vertexIndices,
                screenMesh, OrbitCam.cameraPos,
                128, 72,
                15, 0.005f, 0.35f, true, 0.03f, gDepthScale);

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
                                             128, 72, gDepthScale, params,
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
                                             128, 72, gDepthScale, params,
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

            static Reg3D::BVHTree convergenceBvhTree;
            convergenceBvhTree.build(liverMesh3D->mVertices, liverMesh3D->mIndices);

            Reg3D::RaycastClusterer clusterer(convergenceBvhTree);
            auto clusteringResult = clusterer.performClustering(
                liverMesh3D->mVertices, liverMesh3D->mIndices);

            Reg3DCustom::NoOpen3DRegistration tempReg;
            auto targetCloud = tempReg.extractFrontFacePoints(*screenMesh, 128, 72, gDepthScale);
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
                128, 72, 15, 0.005f, 0.35f, true, 0.03f, gDepthScale);

            std::cout << "=== Registration Complete ===" << std::endl;
        }
        break;

    case GLFW_KEY_F:
        if (currentMainMode == REGISTRATION_MODE) {
            openImageFilePicker();
        }
        break;

    case GLFW_KEY_V:
        g_showClusterVisualization = !g_showClusterVisualization;
        std::cout << "Cluster visualization: "
                  << (g_showClusterVisualization ? "ON" : "OFF") << std::endl;
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
        if(currentMainMode == DEFORM_MODE) {
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
        if(currentMainMode == DEFORM_MODE){
            multiBody->setRigidMode(true);
            deformHandlPlace.state = DeformHandlPlaceData::PLANECUT_MODE;
            std::cout << "Mode: PLANECUT_MODE" << std::endl;
        }

        break;

    case GLFW_KEY_C:
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
            saveARimage = true;
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
        }
        break;

    case GLFW_KEY_TAB:
        if(currentMainMode == REGISTRATION_MODE){
            OrbitCam.switchTarget();
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
        outs.precision(2);
        outs << std::fixed

             << "Window: " << gWindowWidth << "x" << gWindowHeight << "    "
             << "Aspect: " << aspectWidth << ":" << aspectHeight
             << " (" << aspectRatio << ")";
        glfwSetWindowTitle(window, outs.str().c_str());

        frameCount = 0;
    }

    frameCount++;
}
