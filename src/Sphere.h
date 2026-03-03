// Sphere.h
#ifndef SPHERE_H
#define SPHERE_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <cmath>
#include "ShaderProgram.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SphereMesh {
public:
    GLuint VAO = 0, VBO = 0, EBO = 0;
    std::vector<GLfloat> vertices;
    std::vector<GLuint> indices;

    void generate(float radius = 1.0f, int sectors = 20, int stacks = 20);
    void setup();
    void draw(ShaderProgram& shader, const glm::vec3& position,
              const glm::vec3& color, float scale = 0.1f,
              const glm::mat4& view = glm::mat4(1.0f),
              const glm::mat4& projection = glm::mat4(1.0f),
              const glm::vec3& cameraPos = glm::vec3(0.0f));
    void cleanup();
};

// 対応点の色を取得する関数
inline glm::vec3 getPointColor(int index, bool isBright) {
    // 基本色のパレット（最大6点まで対応）
    std::vector<glm::vec3> baseColors = {
        glm::vec3(1.0f, 0.0f, 0.0f),   // 赤
        glm::vec3(0.0f, 1.0f, 0.0f),   // 緑
        glm::vec3(0.0f, 0.0f, 1.0f),   // 青
        glm::vec3(1.0f, 1.0f, 0.0f),   // 黄
        glm::vec3(1.0f, 0.0f, 1.0f),   // マゼンタ
        glm::vec3(0.0f, 1.0f, 1.0f)    // シアン
    };

    if (index >= baseColors.size()) {
        index = index % baseColors.size();
    }

    glm::vec3 color = baseColors[index];

    if (isBright) {
        return color;  // そのまま（明るい）
    } else {
        return color * 0.7f;  // 70%の明度（少し暗い）
    }
}

#endif // SPHERE_H
