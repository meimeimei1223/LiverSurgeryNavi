// Sphere.cpp
#include "Sphere.h"

void SphereMesh::generate(float radius, int sectors, int stacks) {
    vertices.clear();
    indices.clear();
    
    float x, y, z, xy;
    float nx, ny, nz, lengthInv = 1.0f / radius;
    float sectorStep = 2 * M_PI / sectors;
    float stackStep = M_PI / stacks;
    float sectorAngle, stackAngle;
    
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    
    // 頂点生成
    for(int i = 0; i <= stacks; ++i) {
        stackAngle = M_PI / 2 - i * stackStep;
        xy = radius * cosf(stackAngle);
        z = radius * sinf(stackAngle);
        
        for(int j = 0; j <= sectors; ++j) {
            sectorAngle = j * sectorStep;
            x = xy * cosf(sectorAngle);
            y = xy * sinf(sectorAngle);
            positions.push_back(glm::vec3(x, y, z));
            
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            normals.push_back(glm::vec3(nx, ny, nz));
        }
    }
    
    // インデックス生成
    int k1, k2;
    for(int i = 0; i < stacks; ++i) {
        k1 = i * (sectors + 1);
        k2 = k1 + sectors + 1;
        
        for(int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if(i != 0) {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }
            if(i != (stacks-1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }
    
    // インターリーブ形式に変換
    vertices.clear();
    for(size_t i = 0; i < positions.size(); i++) {
        vertices.push_back(positions[i].x);
        vertices.push_back(positions[i].y);
        vertices.push_back(positions[i].z);
        vertices.push_back(normals[i].x);
        vertices.push_back(normals[i].y);
        vertices.push_back(normals[i].z);
    }
}

void SphereMesh::setup() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat),
                 vertices.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat),
                          (void*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
                 indices.data(), GL_STATIC_DRAW);
    
    glBindVertexArray(0);
}

void SphereMesh::draw(ShaderProgram& shader, const glm::vec3& position,
                      const glm::vec3& color, float scale,
                      const glm::mat4& view, const glm::mat4& projection,
                      const glm::vec3& cameraPos) {
    glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f), position);
    modelMatrix = glm::scale(modelMatrix, glm::vec3(scale));
    
    shader.use();
    shader.setUniform("model", modelMatrix);
    shader.setUniform("vertColor", glm::vec4(color, 1.0f));
    shader.setUniform("view", view);
    shader.setUniform("projection", projection);
    shader.setUniform("lightPos", cameraPos);
    shader.setUniform("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));
    shader.setUniform("viewPos", cameraPos);
    shader.setUniform("useTexture", false);
    
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void SphereMesh::cleanup() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}
