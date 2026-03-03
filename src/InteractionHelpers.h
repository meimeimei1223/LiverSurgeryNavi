#pragma once

#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "mCutMesh.h"
#include "MeshDrawing.h"
#include "FullSphereCameraWithTarget.h"
#include "RegistrationUI.h"
#include "DepthRunner.h"

extern int              gWindowWidth, gWindowHeight;
extern bool             splitScreenMode;
extern FullSphereCamera OrbitCam;
extern FullSphereCamera OrbitCamLeft_Target;
extern FullSphereCamera OrbitCamRight_Screen;
extern std::vector<mCutMesh*> allMeshes;
extern RegistrationData registrationHandle;

extern std::vector<DepthRunnerPoint> gUserSegPoints;
extern std::vector<glm::vec3>        gUserSegPoints3D;
extern std::vector<bool>             gUserSegPointsFG;

inline void translateAllMeshes(const glm::vec3& direction) {
    for (auto* m : allMeshes) { m->translate(direction); setUp(*m); }
}

inline void rotateAllMeshes(const glm::vec3& center,
                            const glm::vec3& axisX, float angleX,
                            const glm::vec3& axisY, float angleY) {
    for (auto* m : allMeshes) { m->rotateAround(center, axisX, angleX, axisY, angleY); setUp(*m); }
}

inline void scaleAllMeshes(const glm::vec3& center, float scale) {
    for (auto* m : allMeshes) { m->scaleAround(center, scale); setUp(*m); }
}

inline void scaleRegistrationPoints(const glm::vec3& center, float scale) {
    for (auto& pt : registrationHandle.objectPoints) {
        pt = (pt - center) * scale + center;
    }
}

inline FullSphereCamera* getActiveCamera(GLFWwindow* window) {
    if (!splitScreenMode) return &OrbitCam;
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return (xpos < gWindowWidth / 2.0) ? &OrbitCamLeft_Target : &OrbitCamRight_Screen;
}

inline FullSphereCamera* getActiveCameraWithSide(GLFWwindow* window,
                                                 bool& outIsLeft,
                                                 bool& outIsRight) {
    outIsLeft = false;
    outIsRight = false;
    if (!splitScreenMode) return &OrbitCam;
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    if (xpos < gWindowWidth / 2.0) { outIsLeft = true; return &OrbitCamLeft_Target; }
    else                            { outIsRight = true; return &OrbitCamRight_Screen; }
}

inline void clearSegPoints() {
    gUserSegPoints.clear();
    gUserSegPoints3D.clear();
    gUserSegPointsFG.clear();
}

inline float calcScaleFactor(double deltaY, float scaleSpeed) {
    return (deltaY > 0) ? scaleSpeed : (1.0f / scaleSpeed);
}
