#ifndef MESH_DRAWING_H
#define MESH_DRAWING_H

#include <vector>
#include <algorithm>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include "ShaderProgram.h"
#include "mCutMesh.h"

// [deform-sep Step 2] SoftBody include + forward-decl removed. The
// SoftBody-dependent draw_AllVisMeshesWithExtraMesh() moved to
// deform/src/MeshDrawingSoftBody.h. This header is now mCutMesh-only.

// ============================================================================
// Transparent mesh drawing with per-triangle depth sorting
// ============================================================================

// Single shader version (no texture support) — pointer version to avoid deep copies
inline void draw_AllmCutMeshes(const std::vector<mCutMesh*>& meshes,
                               ShaderProgram& shader,
                               const glm::vec3& camPos,
                               const std::vector<glm::vec4>& meshColors,
                               const glm::mat4& model,
                               const glm::mat4& view,
                               const glm::mat4& projection) {
    shader.use();

    shader.setUniform("model", model);
    shader.setUniform("view", view);
    shader.setUniform("projection", projection);
    shader.setUniform("lightPos", camPos);
    shader.setUniform("viewPos", camPos);
    shader.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shader.setUniform("useTexture", false);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    struct TriangleInfo {
        size_t meshIndex;
        size_t triangleIndex;
        float distance;
    };

    std::vector<TriangleInfo> allTriangles;

    for (size_t meshIdx = 0; meshIdx < meshes.size(); meshIdx++) {
        const auto& mesh = *meshes[meshIdx];

        for (size_t triIdx = 0; triIdx < mesh.mIndices.size() / 3; triIdx++) {
            glm::vec3 center(0.0f);
            for (int v = 0; v < 3; v++) {
                size_t idx = mesh.mIndices[triIdx * 3 + v];
                center.x += mesh.mVertices[idx * 3];
                center.y += mesh.mVertices[idx * 3 + 1];
                center.z += mesh.mVertices[idx * 3 + 2];
            }
            center /= 3.0f;

            glm::vec4 worldCenter = model * glm::vec4(center, 1.0f);
            float distance = glm::length(camPos - glm::vec3(worldCenter));

            allTriangles.push_back({meshIdx, triIdx, distance});
        }
    }

    std::sort(allTriangles.begin(), allTriangles.end(),
              [](const TriangleInfo& a, const TriangleInfo& b) {
                  return a.distance > b.distance;
              });

    GLuint lastVAO = 0;
    glm::vec4 lastColor(-1.0f);

    for (const auto& tri : allTriangles) {
        const auto& mesh = *meshes[tri.meshIndex];

        if (lastVAO != mesh.VAO) {
            glBindVertexArray(mesh.VAO);
            lastVAO = mesh.VAO;
        }

        glm::vec4 currentColor = meshColors[tri.meshIndex % meshColors.size()];
        if (lastColor != currentColor) {
            shader.setUniform("vertColor", currentColor);
            lastColor = currentColor;
        }

        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT,
                       (void*)(tri.triangleIndex * 3 * sizeof(GLuint)));
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
}

// Dual shader version (with texture support for one mesh) — pointer version
inline void draw_AllmCutMeshes(const std::vector<mCutMesh*>& meshes,
                               ShaderProgram& shaderBasic,
                               ShaderProgram& shaderCube,
                               const glm::vec3& camPos,
                               const std::vector<glm::vec4>& meshColors,
                               const glm::mat4& model,
                               const glm::mat4& view,
                               const glm::mat4& projection,
                               int textureMeshIndex = -1) {

    struct TriangleInfo {
        size_t meshIndex;
        size_t triangleIndex;
        float distance;
    };

    std::vector<TriangleInfo> allTriangles;

    for (size_t meshIdx = 0; meshIdx < meshes.size(); meshIdx++) {
        const auto& mesh = *meshes[meshIdx];

        for (size_t triIdx = 0; triIdx < mesh.mIndices.size() / 3; triIdx++) {
            glm::vec3 center(0.0f);
            for (int v = 0; v < 3; v++) {
                size_t idx = mesh.mIndices[triIdx * 3 + v];
                center.x += mesh.mVertices[idx * 3];
                center.y += mesh.mVertices[idx * 3 + 1];
                center.z += mesh.mVertices[idx * 3 + 2];
            }
            center /= 3.0f;

            glm::vec4 worldCenter = model * glm::vec4(center, 1.0f);
            float distance = glm::length(camPos - glm::vec3(worldCenter));

            allTriangles.push_back({meshIdx, triIdx, distance});
        }
    }

    std::sort(allTriangles.begin(), allTriangles.end(),
              [](const TriangleInfo& a, const TriangleInfo& b) {
                  return a.distance > b.distance;
              });

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    GLuint lastVAO = 0;
    glm::vec4 lastColor(-1.0f);
    int lastShaderType = -1;

    for (const auto& tri : allTriangles) {
        const auto& mesh = *meshes[tri.meshIndex];

        int currentShaderType = (tri.meshIndex == textureMeshIndex) ? 1 : 0;

        if (lastShaderType != currentShaderType) {
            if (currentShaderType == 1) {
                shaderCube.use();
                shaderCube.setUniform("model", model);
                shaderCube.setUniform("view", view);
                shaderCube.setUniform("projection", projection);
                shaderCube.setUniform("lightPos", camPos);
                shaderCube.setUniform("viewPos", camPos);
                shaderCube.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
                shaderCube.setUniform("useTexture", true);

                if (mesh.hasTexture) {
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, mesh.textureID);
                    shaderCube.setUniform("texture1", 0);
                }
            } else {
                shaderBasic.use();
                shaderBasic.setUniform("model", model);
                shaderBasic.setUniform("view", view);
                shaderBasic.setUniform("projection", projection);
                shaderBasic.setUniform("lightPos", camPos);
                shaderBasic.setUniform("viewPos", camPos);
                shaderBasic.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));

                glBindTexture(GL_TEXTURE_2D, 0);
            }

            lastShaderType = currentShaderType;
            lastColor = glm::vec4(-1.0f);
        }

        if (lastVAO != mesh.VAO) {
            glBindVertexArray(mesh.VAO);
            lastVAO = mesh.VAO;
        }

        glm::vec4 currentColor = meshColors[tri.meshIndex % meshColors.size()];
        if (lastColor != currentColor) {
            if (currentShaderType == 1) {
                shaderCube.setUniform("vertColor", currentColor);
            } else {
                shaderBasic.setUniform("vertColor", currentColor);
            }
            lastColor = currentColor;
        }

        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT,
                       (void*)(tri.triangleIndex * 3 * sizeof(GLuint)));
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// [deform-sep Step 2] draw_AllVisMeshesWithExtraMesh(SoftBody*, ...) moved to
// deform/src/MeshDrawingSoftBody.h.

#endif // MESH_DRAWING_H
