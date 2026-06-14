// Grabber.h
// マウスでSoftBodyをグラブ／ハンドル配置するためのクラス（単独ヘッダ管理）
//
// 依存(extern グローバル)：
//   gWindowWidth, gWindowHeight, view, projection, isDragging,
//   hit_position, hit_index, bunnyPos
//   → main.cpp で定義する（RayCast.h でも extern 宣言済み）
#pragma once

#include <iostream>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "RayCast.h"     // extern globals(view, projection 等) と RayCast::screenToRay
#include "SoftBody.h"

// bunnyPos は RayCast.h には無いので別途宣言
extern glm::vec3 bunnyPos;

// シーンスケール(DeformGlobals.h で定義, handle 無し時の grabThreshold 用)
extern float gSceneDiag;

// [REG-parity AR] シーン(レターボックス)ビューポート (0,0,w,h)。main.cpp で定義。
//   screenToRay はオフセットを見ないので、呼び出し側がウィンドウ座標を矩形ローカル
//   に変換して渡す前提 (REG の compute3DViewport 運用と同じ)。
extern glm::vec4 g_sceneViewport;

class Grabber {
public:
    Grabber()
        : physicsObject(nullptr),
        grabDistance(0.0f),
        prevPosition(0.0f),
        velocity(0.0f),
        time(0.0f) {}

    void setPhysicsObject(SoftBody* object) { physicsObject = object; }

    void update(float deltaTime) { time += deltaTime; }

    // ハンドル配置(中央に1点を打ってhandleGroup作成)
    void placeSphere(float screenX, float screenY, float groupRadius = 1.0f) {
        if (!physicsObject) return;

        RayCast::Ray worldRay = RayCast::screenToRay(
            screenX, screenY, view, projection,
            g_sceneViewport);

        glm::mat4 modelMatrix    = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);

        RayCast::Ray localRay;
        localRay.origin    = glm::vec3(invModelMatrix * glm::vec4(worldRay.origin, 1.0f));
        localRay.direction = glm::normalize(
            glm::vec3(invModelMatrix * glm::vec4(worldRay.direction, 0.0f)));

        RayCast::RayHit hit = raycastAgainstMesh(localRay);
        if (hit.hit) {
            physicsObject->createHandleGroup(hit.position, groupRadius);

            glm::vec4 worldHitPos = modelMatrix * glm::vec4(hit.position, 1.0f);
            hit_position = glm::vec3(worldHitPos);
            hit_index    = (int)physicsObject->handleGroups.size() - 1;

            std::cout << "[Grabber] handleGroup r=" << groupRadius
                      << " local(" << hit.position.x << "," << hit.position.y << ","
                      << hit.position.z << ")" << std::endl;
        } else {
            hit_index = -1;
        }
        isDragging = false;
    }

    bool hitTest(float screenX, float screenY) {
        if (!physicsObject) return false;
        RayCast::Ray worldRay = RayCast::screenToRay(
            screenX, screenY, view, projection,
            g_sceneViewport);

        glm::mat4 modelMatrix    = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);

        RayCast::Ray localRay;
        localRay.origin    = glm::vec3(invModelMatrix * glm::vec4(worldRay.origin, 1.0f));
        localRay.direction = glm::normalize(
            glm::vec3(invModelMatrix * glm::vec4(worldRay.direction, 0.0f)));

        return raycastAgainstMesh(localRay).hit;
    }

    void startGrab(float screenX, float screenY) {
        if (!physicsObject) return;

        RayCast::Ray worldRay = RayCast::screenToRay(
            screenX, screenY, view, projection,
            g_sceneViewport);

        glm::mat4 modelMatrix    = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);

        RayCast::Ray localRay;
        localRay.origin    = glm::vec3(invModelMatrix * glm::vec4(worldRay.origin, 1.0f));
        localRay.direction = glm::normalize(
            glm::vec3(invModelMatrix * glm::vec4(worldRay.direction, 0.0f)));

        RayCast::RayHit hit = raycastAgainstMesh(localRay);
        if (hit.hit) {
            glm::vec4 worldHitPos = modelMatrix * glm::vec4(hit.position, 1.0f);
            hit_position = glm::vec3(worldHitPos);
            grabDistance = glm::length(hit_position - worldRay.origin);
            prevPosition = hit_position;
            velocity     = glm::vec3(0.0f);
            time         = 0.0f;

            // grabThreshold: handle が無い場合のフォールバック半径。
            // scale=10 時代は 1.0f が「シーンの ~13.6%」だったので、
            // 同じ比率を gSceneDiag からスケールして求める。
            // (gSceneDiag 未初期化時は 1.0f にフォールバック)
            float grabThreshold = (gSceneDiag > 1e-6f)
                                      ? (1.0f * gSceneDiag / 7.36f)   // kRefSceneDiag に揃える
                                      : 1.0f;
            for (const auto& g : physicsObject->handleGroups)
                grabThreshold = std::max(grabThreshold, g.radius);

            physicsObject->smartGrab(hit.position, grabThreshold);
            isDragging = true;
        }
    }

    void moveGrab(float screenX, float screenY, float deltaTime) {
        if (!physicsObject || !isDragging) return;

        RayCast::Ray worldRay = RayCast::screenToRay(
            screenX, screenY, view, projection,
            g_sceneViewport);

        glm::vec3 newPosition = worldRay.origin + worldRay.direction * grabDistance;
        if (time > 0.0f) velocity = (newPosition - prevPosition) / time;
        hit_position = newPosition;

        glm::mat4 modelMatrix    = glm::translate(glm::mat4(1.0f), bunnyPos);
        glm::mat4 invModelMatrix = glm::inverse(modelMatrix);
        glm::vec3 localPos = glm::vec3(invModelMatrix * glm::vec4(newPosition, 1.0f));
        glm::vec3 localVel = glm::vec3(invModelMatrix * glm::vec4(velocity,    0.0f));

        physicsObject->smartMove(localPos, localVel);

        prevPosition = newPosition;
        time         = deltaTime;
    }

    void endGrab() {
        if (physicsObject) {
            glm::mat4 modelMatrix    = glm::translate(glm::mat4(1.0f), bunnyPos);
            glm::mat4 invModelMatrix = glm::inverse(modelMatrix);
            glm::vec3 localPos = glm::vec3(invModelMatrix * glm::vec4(hit_position, 1.0f));
            glm::vec3 localVel = glm::vec3(invModelMatrix * glm::vec4(velocity,    0.0f));
            physicsObject->smartEndGrab(localPos, localVel);
        }
        isDragging = false;
    }

private:
    // [deform-sep Step 1] RayCast lost its SoftBody& overload. Pull the surface
    // positions/indices out of the SoftBody and forward to the generic
    // intersectMesh. getPositions() is vector<float> (== vector<GLfloat>) so it
    // passes directly; tetSurfaceTriIds is vector<int> so convert to GLuint.
    RayCast::RayHit raycastAgainstMesh(const RayCast::Ray& localRay) const {
        const std::vector<float>& posF = physicsObject->getPositions();
        const std::vector<int>&   triI = physicsObject->getMeshData().tetSurfaceTriIds;
        std::vector<GLuint> triU(triI.begin(), triI.end());
        return RayCast::intersectMesh(localRay, posF, triU);
    }

    SoftBody* physicsObject;
    float     grabDistance;
    glm::vec3 prevPosition;
    glm::vec3 velocity;
    float     time;
};
