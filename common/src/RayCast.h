#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "SoftBody.h"
#include "mCutMesh.h"
#include "FullSphereCameraWithTarget.h"

// =========================================================================
//  extern globals (defined in main.cpp)
// =========================================================================
extern int          gWindowWidth, gWindowHeight;
extern glm::mat4    model, view, projection;
extern glm::vec3    objPos;
extern glm::vec3    hit_position;
extern int          hit_index;
extern bool         isDragging;
extern FullSphereCamera OrbitCam;

// =========================================================================
//  RayCast class
// =========================================================================
class RayCast {
public:
    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;
    };

    struct RayHit {
        bool hit;
        float distance;
        glm::vec3 position;
        SoftBody* hitObject;
    };

    struct RayHitTri {
        bool hit;
        float distance;
        glm::vec3 position;
    };

    static Ray screenToRay(float screenX, float screenY,
                           const glm::mat4& view,
                           const glm::mat4& projection,
                           const glm::vec4& viewport) {

        float ndcX = (2.0f * screenX) / viewport.z - 1.0f;
        float ndcY = 1.0f - (2.0f * screenY) / viewport.w;

        glm::vec4 nearPoint = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
        glm::vec4 farPoint = glm::vec4(ndcX, ndcY, 1.0f, 1.0f);

        glm::mat4 invVP = glm::inverse(projection * view);

        glm::vec4 worldNear = invVP * nearPoint;
        glm::vec4 worldFar = invVP * farPoint;

        worldNear /= worldNear.w;
        worldFar /= worldFar.w;

        Ray ray;
        ray.origin = glm::vec3(worldNear);
        ray.direction = glm::normalize(glm::vec3(worldFar - worldNear));

        return ray;
    }

    static RayHit intersectMesh(const Ray& ray, SoftBody& mesh) {
        RayHit result = { false, std::numeric_limits<float>::max(), glm::vec3(0), nullptr };

        const auto& positions = mesh.getPositions();
        const auto& surfaceTriIds = mesh.getMeshData().tetSurfaceTriIds;

        std::cout << "Ray origin: " << ray.origin.x << ", " << ray.origin.y << ", " << ray.origin.z << std::endl;
        std::cout << "Ray direction: " << ray.direction.x << ", " << ray.direction.y << ", " << ray.direction.z << std::endl;

        float t, u, v;
        for (size_t i = 0; i < surfaceTriIds.size(); i += 3) {
            int idx1 = surfaceTriIds[i];
            int idx2 = surfaceTriIds[i + 1];
            int idx3 = surfaceTriIds[i + 2];

            glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
            glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
            glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);

            if (rayTriangleIntersect(ray.origin, ray.direction, v1, v2, v3, t, u, v)) {
                if (t < result.distance) {
                    result.hit = true;
                    result.distance = t;
                    result.position = ray.origin + ray.direction * t;
                    result.hitObject = &mesh;
                    std::cout << "Hit at triangle " << i/3 << std::endl;
                    std::cout << "Hit position: " << result.position.x << ", "
                              << result.position.y << ", " << result.position.z << std::endl;
                }
            }
        }

        return result;
    }

    static RayHitTri intersectMesh(const Ray& ray, std::vector<GLfloat> vertices, std::vector<GLuint> indices) {
        RayHitTri result = { false, std::numeric_limits<float>::max(), glm::vec3(0)};

        const auto& positions = vertices;
        const auto& surfaceTriIds = indices;

        std::cout << "Ray origin: " << ray.origin.x << ", " << ray.origin.y << ", " << ray.origin.z << std::endl;
        std::cout << "Ray direction: " << ray.direction.x << ", " << ray.direction.y << ", " << ray.direction.z << std::endl;

        float t, u, v;
        for (size_t i = 0; i < surfaceTriIds.size(); i += 3) {
            int idx1 = surfaceTriIds[i];
            int idx2 = surfaceTriIds[i + 1];
            int idx3 = surfaceTriIds[i + 2];

            glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
            glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
            glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);

            if (rayTriangleIntersect(ray.origin, ray.direction, v1, v2, v3, t, u, v)) {
                if (t < result.distance) {
                    result.hit = true;
                    result.distance = t;
                    result.position = ray.origin + ray.direction * t;
                    hit_index = i/3;
                    std::cout << "Hit at triangle " << i/3 << std::endl;
                    std::cout << "Hit position: " << result.position.x << ", "
                              << result.position.y << ", " << result.position.z << std::endl;
                }
            }
        }
        return result;
    }

private:

    static bool rayTriangleIntersect(
        const glm::vec3& rayOrigin,
        const glm::vec3& rayDir,
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        float& t,
        float& u,
        float& v) {

        const float EPSILON = 0.0000001f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(rayDir, edge2);
        float a = glm::dot(edge1, h);

        if (a > -EPSILON && a < EPSILON) return false;

        float f = 1.0f / a;
        glm::vec3 s = rayOrigin - v0;
        u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        glm::vec3 q = glm::cross(s, edge1);
        v = f * glm::dot(rayDir, q);

        if (v < 0.0f || u + v > 1.0f) return false;

        t = f * glm::dot(edge2, q);
        return t > EPSILON;
    }
};

// =========================================================================
//  createProjectionForViewport  (used by both Umeyama & Depth paths)
// =========================================================================
inline glm::mat4 createProjectionForViewport(int viewportWidth, int viewportHeight,
                                             const FullSphereCamera& cam) {
    glm::mat4 proj(0.0f);

    float fx_adjusted = cam.fx;
    float fy_adjusted = cam.fy;

    float cx_viewport = viewportWidth / 2.0f;
    float cy_viewport = viewportHeight / 2.0f;

    proj[0][0] = 2.0f * fx_adjusted / viewportWidth;
    proj[1][1] = 2.0f * fy_adjusted / viewportHeight;
    proj[2][0] = 1.0f - 2.0f * cx_viewport / viewportWidth;
    proj[2][1] = 1.0f - 2.0f * cy_viewport / viewportHeight;
    proj[2][2] = -(cam.farPlane + cam.nearPlane) / (cam.farPlane - cam.nearPlane);
    proj[2][3] = -1.0f;
    proj[3][2] = -2.0f * cam.farPlane * cam.nearPlane / (cam.farPlane - cam.nearPlane);

    return proj;
}

// =========================================================================
//  FindHit  (generic vertex/index hit test)
// =========================================================================
inline void FindHit(float screenX, float screenY,
                    const std::vector<GLfloat> vertices,
                    const std::vector<GLuint> indices) {

    OrbitCam.UpdateCamera();
    view = OrbitCam.view;
    projection = OrbitCam.projection;

    RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection,
                                                 glm::vec4(0, 0, gWindowWidth, gWindowHeight));

    glm::mat4 modelMatrix = glm::mat4(1.0f);

    RayCast::RayHitTri hit = RayCast::intersectMesh(worldRay, vertices, indices);

    std::cout << "Hit test with ray: origin=("
              << worldRay.origin.x << "," << worldRay.origin.y << "," << worldRay.origin.z
              << "), dir=(" << worldRay.direction.x << "," << worldRay.direction.y
              << "," << worldRay.direction.z << ")" << std::endl;

    std::cout << "Hit: " << (hit.hit ? "Yes" : "No") << std::endl;

    if (hit.hit) {
        std::cout << "Hit distance: " << hit.distance << std::endl;
        std::cout << "Hit position: " << hit.position.x << ", "
                  << hit.position.y << ", " << hit.position.z << std::endl;

        hit_position = hit.position;
        isDragging = true;
    } else {
        hit_index = -1;
        isDragging = false;
    }

    std::cout << "hit_index: " << hit_index << std::endl;
}

// =========================================================================
//  FindHitWithCamera  (vertices + indices version)
// =========================================================================
inline void FindHitWithCamera(float screenX, float screenY,
                              const std::vector<GLfloat>& vertices,
                              const std::vector<GLuint>& indices,
                              FullSphereCamera* camera,
                              int viewportWidth, int viewportHeight) {

    camera->UpdateCamera();

    projection = createProjectionForViewport(viewportWidth, viewportHeight, *camera);

    glm::vec4 viewport(0, 0, viewportWidth, viewportHeight);

    RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection, viewport);

    RayCast::RayHitTri hit = RayCast::intersectMesh(worldRay, vertices, indices);

    if (hit.hit) {
        glm::vec3 hitPos = hit.position;
        float minD2 = std::numeric_limits<float>::max();
        int closestVertexIndex = -1;

        for (size_t i = 0; i < vertices.size() / 3; i++) {
            glm::vec3 vertexPos(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
            glm::vec3 diff = vertexPos - hitPos;
            float d2 = glm::dot(diff, diff);

            if (d2 < minD2) {
                minD2 = d2;
                closestVertexIndex = i;
            }
        }

        if (closestVertexIndex >= 0) {
            hit_position = glm::vec3(
                vertices[closestVertexIndex * 3],
                vertices[closestVertexIndex * 3 + 1],
                vertices[closestVertexIndex * 3 + 2]
                );
            hit_index = closestVertexIndex;
        }
        isDragging = true;
    } else {
        hit_index = -1;
        isDragging = false;
    }
}

// =========================================================================
//  FindHitWithCameraMultipleMeshes
// =========================================================================
inline void FindHitWithCameraMultipleMeshes(
    float screenX, float screenY,
    const std::vector<mCutMesh*>& meshes,
    FullSphereCamera* camera,
    int viewportWidth, int viewportHeight) {

    camera->UpdateCamera();
    projection = createProjectionForViewport(viewportWidth, viewportHeight, *camera);
    glm::vec4 viewport(0, 0, viewportWidth, viewportHeight);

    RayCast::Ray worldRay = RayCast::screenToRay(screenX, screenY, view, projection, viewport);

    struct HitResult {
        bool hit = false;
        float distance = std::numeric_limits<float>::max();
        glm::vec3 position;
        int meshIndex = -1;
        int vertexIndex = -1;
        mCutMesh* mesh = nullptr;
    } closestHit;

    for (size_t meshIdx = 0; meshIdx < meshes.size(); meshIdx++) {
        if (!meshes[meshIdx] || meshes[meshIdx]->mVertices.empty()) continue;

        RayCast::RayHitTri hit = RayCast::intersectMesh(
            worldRay,
            meshes[meshIdx]->mVertices,
            meshes[meshIdx]->mIndices);

        if (hit.hit && hit.distance < closestHit.distance) {
            closestHit.hit = true;
            closestHit.distance = hit.distance;
            closestHit.position = hit.position;
            closestHit.meshIndex = meshIdx;
            closestHit.mesh = meshes[meshIdx];

            float minD2 = std::numeric_limits<float>::max();
            for (size_t i = 0; i < meshes[meshIdx]->mVertices.size() / 3; i++) {
                glm::vec3 vertexPos(
                    meshes[meshIdx]->mVertices[i * 3],
                    meshes[meshIdx]->mVertices[i * 3 + 1],
                    meshes[meshIdx]->mVertices[i * 3 + 2]);
                glm::vec3 diff = vertexPos - hit.position;
                float d2 = glm::dot(diff, diff);

                if (d2 < minD2) {
                    minD2 = d2;
                    closestHit.vertexIndex = i;
                }
            }
        }
    }

    if (closestHit.hit) {
        if (closestHit.vertexIndex >= 0 && closestHit.mesh) {
            hit_position = glm::vec3(
                closestHit.mesh->mVertices[closestHit.vertexIndex * 3],
                closestHit.mesh->mVertices[closestHit.vertexIndex * 3 + 1],
                closestHit.mesh->mVertices[closestHit.vertexIndex * 3 + 2]
                );
            hit_index = closestHit.vertexIndex;

            std::cout << "Hit mesh index: " << closestHit.meshIndex
                      << " at distance: " << closestHit.distance << std::endl;
        }
        isDragging = true;
    } else {
        hit_index = -1;
        isDragging = false;
    }
}

// =========================================================================
//  FindHitWithCamera  (mCutMesh* convenience overload)
// =========================================================================
inline void FindHitWithCamera(float screenX, float screenY,
                              mCutMesh* singleMesh,
                              FullSphereCamera* camera,
                              int viewportWidth, int viewportHeight) {

    if (!singleMesh) return;

    FindHitWithCamera(screenX, screenY,
                      singleMesh->mVertices,
                      singleMesh->mIndices,
                      camera, viewportWidth, viewportHeight);
}
