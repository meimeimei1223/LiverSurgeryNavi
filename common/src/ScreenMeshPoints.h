#pragma once
// =========================================================================
//  ScreenMeshPointCache - draw a heavy mesh as a sparse GL_POINTS cloud.
// -------------------------------------------------------------------------
//  Shared by REGISTRATION (screenMesh) and DEFORM (gTargetMesh): the target
//  point cloud has hundreds of thousands of triangles and is far too heavy to
//  rasterize every frame, so we draw it as points instead.
//
//  Pre-shuffles the vertex indices [0..N-1] into a private EBO, then draws the
//  first `densityPct`% via glDrawElements(GL_POINTS, ...). The shuffle is done
//  once (until the mesh changes or Reshuffle is requested) so moving the
//  density slider does not make points flicker -- it always draws the same
//  shuffled prefix.
//
//  A dedicated VAO is required: binding GL_ELEMENT_ARRAY_BUFFER onto the
//  mesh's own VAO would corrupt its state and break the next triangle draw.
//  Positions/normals are shared from the mesh's VBO/NBO; only the index buffer
//  is owned here. cachedVBO detects setUp()-induced VBO regeneration and
//  rebuilds automatically.
// =========================================================================

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdint>

#include "mCutMesh.h"
#include "ShaderProgram.h"

struct ScreenMeshPointCache {
    GLuint vao            = 0;
    GLuint ebo            = 0;
    size_t totalVerts     = 0;
    GLuint cachedVBO      = 0;     // mesh->VBO id at last build (invalidation key)
    bool   needsReshuffle = false; // user pressed Reshuffle button

    bool ensure(mCutMesh* mesh) {
        if (!mesh || mesh->VBO == 0 || mesh->mVertices.empty()) return false;
        const size_t n = mesh->mVertices.size() / 3;

        // already up-to-date?
        if (vao != 0
            && totalVerts == n
            && cachedVBO == mesh->VBO
            && !needsReshuffle) {
            return true;
        }

        cleanup();

        // shuffled indices [0..n-1]
        std::vector<GLuint> idx(n);
        std::iota(idx.begin(), idx.end(), 0u);
        // 固定シードで再現性確保 (Reshuffle ボタン押下時のみ別シード)
        static uint64_t seed = 0xC0FFEEULL;
        if (needsReshuffle) seed += 1;
        std::mt19937 rng(static_cast<unsigned>(seed));
        std::shuffle(idx.begin(), idx.end(), rng);

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &ebo);
        glBindVertexArray(vao);

        // share mesh->VBO for positions (location=0)
        glBindBuffer(GL_ARRAY_BUFFER, mesh->VBO);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(0);

        // share mesh->NBO for normals (location=1) so the basic shader's
        // lighting calc still gets sensible values
        if (mesh->NBO != 0) {
            glBindBuffer(GL_ARRAY_BUFFER, mesh->NBO);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(1);
        }

        // upload shuffled index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     n * sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);

        totalVerts     = n;
        cachedVBO      = mesh->VBO;
        needsReshuffle = false;
        return true;
    }

    void requestReshuffle() { needsReshuffle = true; }

    void cleanup() {
        if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
        if (ebo) { glDeleteBuffers(1, &ebo);      ebo = 0; }
        totalVerts = 0;
        cachedVBO  = 0;
    }

    // Draw `densityPct`% of the mesh's vertices as GL_POINTS using the basic
    // shader (single color via vertColor, useTexture=false). Lighting uniforms
    // are set so the shader still produces sensible shading.
    void draw(mCutMesh* mesh,
              ShaderProgram& shader,
              const glm::mat4& model,
              const glm::mat4& view,
              const glm::mat4& projection,
              const glm::vec3& camPos,
              const glm::vec4& color,
              float pointSize,
              float densityPct) {
        if (!mesh || mesh->VAO == 0 || mesh->mVertices.empty()) return;
        if (!ensure(mesh)) return;

        shader.use();
        shader.setUniform("model",      model);
        shader.setUniform("view",       view);
        shader.setUniform("projection", projection);
        shader.setUniform("vertColor",  color);
        shader.setUniform("lightPos",   camPos);
        shader.setUniform("viewPos",    camPos);
        shader.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
        shader.setUniform("useTexture", false);

        // density [%] -> draw count (front of shuffled list)
        const float pct = std::clamp(densityPct, 0.1f, 100.0f);
        GLsizei drawCount = (GLsizei)std::max<size_t>(
            1, (size_t)((double)totalVerts * (double)pct / 100.0));

        glPointSize(pointSize);
        glBindVertexArray(vao);
        glDrawElements(GL_POINTS, drawCount, GL_UNSIGNED_INT, (void*)0);
        glBindVertexArray(0);
    }
};
