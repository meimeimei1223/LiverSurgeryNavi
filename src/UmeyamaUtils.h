#pragma once

#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "RayCast.h"
#include "MeshDrawing.h"
#include "RegistrationUI.h"
#include "Sphere.h"

// =========================================================================
//  extern globals (defined in main.cpp)
// =========================================================================
extern int              gWindowWidth, gWindowHeight;
extern glm::mat4        model, view, projection;
extern glm::vec3        objPos;
extern glm::vec3        hit_position;
extern int              hit_index;
extern bool             isDragging;

extern FullSphereCamera OrbitCam;
extern FullSphereCamera OrbitCamLeft_Target;
extern FullSphereCamera OrbitCamRight_Screen;
extern bool             splitScreenMode;

extern mCutMesh*        screenMesh;
extern mCutMesh*        liverMesh3D;
extern mCutMesh*        portalMesh3D;
extern mCutMesh*        veinMesh3D;
extern mCutMesh*        tumorMesh3D;
extern mCutMesh*        segmentMesh3D;
extern mCutMesh*        gbMesh3D;

extern std::vector<float> meshAlphaValues;
extern RegistrationData   registrationHandle;
extern SphereMesh         registrationSphereMarker;

extern std::vector<glm::vec3> gUserSegPoints3D;
extern std::vector<bool>      gUserSegPointsFG;

extern bool g_showCorrespondencePoints;

// =========================================================================
//  pickPointOnBoard
// =========================================================================
inline glm::vec3 pickPointOnBoard(float screenX, float screenY) {

    view = glm::lookAt(OrbitCam.cameraPos, OrbitCam.cameraTarget, OrbitCam.worldUp);
    projection = OrbitCam.createProjectionFromIntrinsics();

    RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection,
                                                 glm::vec4(0, 0, gWindowWidth, gWindowHeight));

    RayCast::RayHitTri hit = RayCast::intersectMesh(worldRay, screenMesh->mVertices, screenMesh->mIndices);

    if (hit.hit) {
        std::cout << "Board hit at distance: " << hit.distance << std::endl;
        return hit.position;
    }

    std::cout << "No hit on texture board" << std::endl;
    return glm::vec3(-999);
}

// =========================================================================
//  pickPointOnBoardWithCamera
// =========================================================================
inline glm::vec3 pickPointOnBoardWithCamera(float screenX, float screenY,
                                            FullSphereCamera* camera,
                                            int viewportWidth, int viewportHeight) {

    view = glm::lookAt(camera->cameraPos, camera->cameraTarget, camera->worldUp);
    projection = createProjectionForViewport(viewportWidth, viewportHeight, *camera);

    glm::vec4 viewport(0, 0, viewportWidth, viewportHeight);

    RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection, viewport);
    RayCast::RayHitTri hit = RayCast::intersectMesh(worldRay, screenMesh->mVertices, screenMesh->mIndices);

    if (hit.hit) {
        return hit.position;
    }

    return glm::vec3(-999);
}

// =========================================================================
//  renderSplitScreen  (Umeyama registration mode)
// =========================================================================
inline void renderSplitScreen(ShaderProgram& shaderProgram, ShaderProgram& shaderProgramCube) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int halfWidth = gWindowWidth / 2;

    // --- Left viewport (liver / object) ---
    glViewport(0, 0, halfWidth, gWindowHeight);

    OrbitCamLeft_Target.currentTarget = TARGET_LIVER;
    OrbitCamLeft_Target.UpdateCamera();

    projection = createProjectionForViewport(halfWidth, gWindowHeight, OrbitCamLeft_Target);
    model = glm::translate(glm::mat4(1.0f), objPos);

    shaderProgram.use();
    shaderProgram.setUniform("model", model);
    shaderProgram.setUniform("view", view);
    shaderProgram.setUniform("projection", projection);
    shaderProgram.setUniform("lightPos", OrbitCamLeft_Target.cameraPos);
    shaderProgram.setUniform("viewPos", OrbitCamLeft_Target.cameraPos);
    shaderProgram.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shaderProgram.setUniform("useTexture", false);

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
        tumorMesh3D, segmentMesh3D, gbMesh3D
    };

    draw_AllmCutMeshes(meshesToDraw, shaderProgram, shaderProgramCube,
                       OrbitCamLeft_Target.cameraPos, customColors,
                       model, view, projection);

    {
        bool activeSelection = (registrationHandle.state == RegistrationData::SELECTING_BOARD_POINTS ||
                                registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS ||
                                registrationHandle.state == RegistrationData::READY_TO_REGISTER);
        if (activeSelection || g_showCorrespondencePoints) {
            for (size_t i = 0; i < registrationHandle.objectPoints.size(); i++) {
                glm::vec3 color = getPointColor(i, false);

                glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), registrationHandle.objectPoints[i]);
                sphereModel = glm::scale(sphereModel, glm::vec3(0.3f));

                shaderProgram.setUniform("model", sphereModel);
                shaderProgram.setUniform("vertColor", glm::vec4(color, 1.0f));

                glBindVertexArray(registrationSphereMarker.VAO);
                glDrawElements(GL_TRIANGLES, registrationSphereMarker.indices.size(), GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            }
        }
    }

    // --- Right viewport (texture board) ---
    glViewport(halfWidth, 0, halfWidth, gWindowHeight);

    OrbitCamRight_Screen.currentTarget = TARGET_TEXTURE;
    OrbitCamRight_Screen.UpdateCamera();

    projection = createProjectionForViewport(halfWidth, gWindowHeight, OrbitCamRight_Screen);
    model = glm::translate(glm::mat4(1.0f), objPos);

    shaderProgramCube.use();
    shaderProgramCube.setUniform("model", model);
    shaderProgramCube.setUniform("view", view);
    shaderProgramCube.setUniform("projection", projection);
    shaderProgramCube.setUniform("lightPos", OrbitCamRight_Screen.cameraPos);
    shaderProgramCube.setUniform("viewPos", OrbitCamRight_Screen.cameraPos);
    shaderProgramCube.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shaderProgramCube.setUniform("vertColor", glm::vec4(screenMesh->mColor, 0.7f));

    if (screenMesh->hasTexture) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, screenMesh->textureID);
        shaderProgramCube.setUniform("texture1", 0);
        shaderProgramCube.setUniform("useTexture", true);
    } else {
        shaderProgramCube.setUniform("useTexture", false);
    }

    glBindVertexArray(screenMesh->VAO);
    glDrawElements(GL_TRIANGLES, screenMesh->mIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    if (screenMesh->hasTexture) {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    {
        bool activeSelection = (registrationHandle.state == RegistrationData::SELECTING_BOARD_POINTS ||
                                registrationHandle.state == RegistrationData::SELECTING_OBJECT_POINTS ||
                                registrationHandle.state == RegistrationData::READY_TO_REGISTER);
        if (activeSelection || g_showCorrespondencePoints) {
            shaderProgram.use();
            for (size_t i = 0; i < registrationHandle.boardPoints.size(); i++) {
                glm::vec3 color = getPointColor(i, true);

                glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), registrationHandle.boardPoints[i]);
                sphereModel = glm::scale(sphereModel, glm::vec3(0.3f));

                shaderProgram.setUniform("model", sphereModel);
                shaderProgram.setUniform("view", view);
                shaderProgram.setUniform("projection", projection);
                shaderProgram.setUniform("lightPos", OrbitCamRight_Screen.cameraPos);
                shaderProgram.setUniform("viewPos", OrbitCamRight_Screen.cameraPos);
                shaderProgram.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
                shaderProgram.setUniform("vertColor", glm::vec4(color, 1.0f));
                shaderProgram.setUniform("useTexture", false);

                glBindVertexArray(registrationSphereMarker.VAO);
                glDrawElements(GL_TRIANGLES, registrationSphereMarker.indices.size(), GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            }
        }
    }

    glViewport(0, 0, gWindowWidth, gWindowHeight);
}

// =========================================================================
//  renderDepthSplitScreen
// =========================================================================
inline void renderDepthSplitScreen(ShaderProgram& shaderProgram, ShaderProgram& shaderProgramCube) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int halfWidth = gWindowWidth / 2;

    // --- Left viewport (3D liver model) ---
    glViewport(0, 0, halfWidth, gWindowHeight);

    OrbitCamLeft_Target.currentTarget = TARGET_LIVER;
    OrbitCamLeft_Target.UpdateCamera();

    projection = createProjectionForViewport(halfWidth, gWindowHeight, OrbitCamLeft_Target);
    model = glm::translate(glm::mat4(1.0f), objPos);

    shaderProgram.use();
    shaderProgram.setUniform("model", model);
    shaderProgram.setUniform("view", view);
    shaderProgram.setUniform("projection", projection);
    shaderProgram.setUniform("lightPos", OrbitCamLeft_Target.cameraPos);
    shaderProgram.setUniform("viewPos", OrbitCamLeft_Target.cameraPos);
    shaderProgram.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shaderProgram.setUniform("useTexture", false);

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
        tumorMesh3D, segmentMesh3D, gbMesh3D
    };

    draw_AllmCutMeshes(meshesToDraw, shaderProgram, shaderProgramCube,
                       OrbitCamLeft_Target.cameraPos, customColors,
                       model, view, projection);

    // --- Right viewport (texture board + segmentation points) ---
    glViewport(halfWidth, 0, halfWidth, gWindowHeight);

    OrbitCamRight_Screen.currentTarget = TARGET_TEXTURE;
    OrbitCamRight_Screen.UpdateCamera();

    projection = createProjectionForViewport(halfWidth, gWindowHeight, OrbitCamRight_Screen);
    model = glm::translate(glm::mat4(1.0f), objPos);

    // Draw texture board
    shaderProgramCube.use();
    shaderProgramCube.setUniform("model", model);
    shaderProgramCube.setUniform("view", view);
    shaderProgramCube.setUniform("projection", projection);
    shaderProgramCube.setUniform("lightPos", OrbitCamRight_Screen.cameraPos);
    shaderProgramCube.setUniform("viewPos", OrbitCamRight_Screen.cameraPos);
    shaderProgramCube.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shaderProgramCube.setUniform("vertColor", glm::vec4(screenMesh->mColor, 0.7f));

    if (screenMesh->hasTexture) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, screenMesh->textureID);
        shaderProgramCube.setUniform("texture1", 0);
        shaderProgramCube.setUniform("useTexture", true);
    } else {
        shaderProgramCube.setUniform("useTexture", false);
    }

    glBindVertexArray(screenMesh->VAO);
    glDrawElements(GL_TRIANGLES, screenMesh->mIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    if (screenMesh->hasTexture) {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Draw segmentation points on RIGHT viewport (green=FG, red=BG)
    if (!gUserSegPoints3D.empty()) {
        shaderProgram.use();
        for (size_t i = 0; i < gUserSegPoints3D.size(); i++) {
            glm::vec3 color = gUserSegPointsFG[i]
                                  ? glm::vec3(0.0f, 1.0f, 0.0f)   // FG = green
                                  : glm::vec3(1.0f, 0.0f, 0.0f);  // BG = red

            glm::mat4 sphereModel = glm::translate(glm::mat4(1.0f), gUserSegPoints3D[i]);
            sphereModel = glm::scale(sphereModel, glm::vec3(0.15f));

            shaderProgram.setUniform("model", sphereModel);
            shaderProgram.setUniform("view", view);
            shaderProgram.setUniform("projection", projection);
            shaderProgram.setUniform("lightPos", OrbitCamRight_Screen.cameraPos);
            shaderProgram.setUniform("viewPos", OrbitCamRight_Screen.cameraPos);
            shaderProgram.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
            shaderProgram.setUniform("vertColor", glm::vec4(color, 1.0f));
            shaderProgram.setUniform("useTexture", false);

            glBindVertexArray(registrationSphereMarker.VAO);
            glDrawElements(GL_TRIANGLES, registrationSphereMarker.indices.size(), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }

    // Reset viewport
    glViewport(0, 0, gWindowWidth, gWindowHeight);
}

// =========================================================================
//  resetRegistrationState
// =========================================================================
inline void resetRegistrationState() {
    splitScreenMode = false;
    registrationHandle.reset();
    registrationHandle.state = RegistrationData::IDLE;
    OrbitCam.cx = gWindowWidth / 2.0f;
    OrbitCam.cy = gWindowHeight / 2.0f;
}

// =========================================================================
//  saveScreenshot
// =========================================================================
//#include "stb_image_write.h"
inline void saveScreenshot(const std::string& filename) {
    OrbitCam.cx = gWindowWidth / 2.0f;
    OrbitCam.cy = gWindowHeight / 2.0f;

    float scale = 10.0f;
    float boardHeight = scale;
    float halfFOVy = atan(gWindowHeight / (2.0f * OrbitCam.fy));
    float requiredDistance = (boardHeight / 2.0f) / tan(halfFOVy);

    OrbitCam.gRadius = requiredDistance;
    OrbitCam.currentTarget = TARGET_TEXTURE;

    std::vector<unsigned char> pixels(gWindowWidth * gWindowHeight * 3);
    glReadPixels(0, 0, gWindowWidth, gWindowHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    int stride = gWindowWidth * 3;
    std::vector<unsigned char> flipped(pixels.size());
    for (int y = 0; y < gWindowHeight; y++) {
        memcpy(&flipped[y * stride],
               &pixels[(gWindowHeight - 1 - y) * stride],
               stride);
    }

    stbi_write_png(filename.c_str(), gWindowWidth, gWindowHeight, 3, flipped.data(), stride);
    std::cout << "Screenshot saved: " << filename << std::endl;
}
