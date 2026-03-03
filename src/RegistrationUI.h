#pragma once

#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "mCutMesh.h"
#include "RegistrationCore.h"

inline void applyRegistrationToMesh(
    RegistrationData& reg,
    std::vector<mCutMesh*>& meshes)
{
    if (!reg.useRegistration) {
        std::cout << "No registration matrix available" << std::endl;
        return;
    }

    for (auto* mesh : meshes) {
        if (mesh && !mesh->mVertices.empty()) {
            for (size_t i = 0; i < mesh->mVertices.size(); i += 3) {
                glm::vec4 vertex(
                    mesh->mVertices[i],
                    mesh->mVertices[i + 1],
                    mesh->mVertices[i + 2],
                    1.0f
                    );

                vertex = reg.registrationMatrix * vertex;

                mesh->mVertices[i] = vertex.x;
                mesh->mVertices[i + 1] = vertex.y;
                mesh->mVertices[i + 2] = vertex.z;
            }
            setUp(*mesh);
        }
    }

    for (auto& point : reg.objectPoints) {
        glm::vec4 p(point, 1.0f);
        p = reg.registrationMatrix * p;
        point = glm::vec3(p);
    }

    std::cout << "Registration transform applied to all meshes" << std::endl;
}

inline void performRegistrationUmeyama(
    RegistrationData& reg,
    std::vector<mCutMesh*>& meshes)
{
    if (!reg.canRegister()) return;

    glm::mat4 T = Reg3D::UmeyamaRegistration(
        reg.objectPoints,
        reg.boardPoints
        );

    reg.registrationMatrix = T;
    reg.useRegistration = true;
    reg.state = RegistrationData::REGISTERED;
    reg.scaleFactor = glm::length(glm::vec3(T[0]));

    float totalError = 0;
    float sumSqError = 0;
    float maxErr = 0;
    for (size_t i = 0; i < reg.objectPoints.size(); i++) {
        glm::vec4 transformed = T * glm::vec4(reg.objectPoints[i], 1.0f);
        float error = glm::distance(
            glm::vec3(transformed),
            reg.boardPoints[i]
            );
        totalError += error;
        sumSqError += error * error;
        if (error > maxErr) maxErr = error;
    }
    float n = (float)reg.objectPoints.size();
    reg.averageError = totalError / n;
    reg.rmse = std::sqrt(sumSqError / n);
    reg.maxError = maxErr;

    std::cout << "Scale factor: " << reg.scaleFactor << std::endl;
    std::cout << "Average error: " << reg.averageError << " mm" << std::endl;
    std::cout << "RMSE: " << reg.rmse << " mm" << std::endl;
    std::cout << "Max error: " << reg.maxError << " mm" << std::endl;

    applyRegistrationToMesh(reg, meshes);
}
