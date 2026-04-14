#include "SoftBody.h"
#include "VectorMath.h"
#include <iostream>
#include "ShaderProgram.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "SoftBody.h"
#include <algorithm>
#include <limits>

void SoftBody::setRigidMode(bool enabled) {
    rigidMode = enabled;
    if (enabled) {
        glm::vec3 centerOfMass(0.0f);
        for (size_t i = 0; i < numParticles; i++) {
            centerOfMass.x += positions[i * 3];
            centerOfMass.y += positions[i * 3 + 1];
            centerOfMass.z += positions[i * 3 + 2];
        }
        centerOfMass /= float(numParticles);
        rigidPosition = centerOfMass;

        restPositionsRelative.clear();
        restPositionsRelative.reserve(numParticles);
        for (size_t i = 0; i < numParticles; i++) {
            glm::vec3 relPos;
            relPos.x = positions[i * 3] - centerOfMass.x;
            relPos.y = positions[i * 3 + 1] - centerOfMass.y;
            relPos.z = positions[i * 3 + 2] - centerOfMass.z;
            restPositionsRelative.push_back(relPos);
        }

        rigidRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        rigidVelocity = glm::vec3(0.0f);
        rigidAngularVelocity = glm::vec3(0.0f);
    } else {
        std::fill(velocities.begin(), velocities.end(), 0.0f);
        prevPositions = positions;
        std::fill(edgeLambdas.begin(), edgeLambdas.end(), 0.0f);
        std::fill(volLambdas.begin(), volLambdas.end(), 0.0f);
    }
}

void SoftBody::updateRigidTransform() {
    glm::mat3 rotMatrix = glm::mat3_cast(rigidRotation);

    for (size_t i = 0; i < numParticles; i++) {
        glm::vec3 rotatedPos = rotMatrix * restPositionsRelative[i];
        positions[i * 3] = rigidPosition.x + rotatedPos.x;
        positions[i * 3 + 1] = rigidPosition.y + rotatedPos.y;
        positions[i * 3 + 2] = rigidPosition.z + rotatedPos.z;
    }
}

void SoftBody::rigidTranslate(const glm::vec3& delta) {
    if (!rigidMode) return;
    rigidPosition += delta;
    updateRigidTransform();
}

void SoftBody::rigidRotateAroundCenter(const glm::vec3& axis, float angle) {
    if (!rigidMode) return;
    if (glm::length(axis) < 1e-6f) return;
    glm::quat deltaRot = glm::angleAxis(angle, glm::normalize(axis));
    rigidRotation = deltaRot * rigidRotation;
    rigidRotation = glm::normalize(rigidRotation);
    updateRigidTransform();
}

void SoftBody::preSolve(float dt, const glm::vec3& gravity) {
    if (rigidMode) {
        if (!isDragging) {
            rigidVelocity += gravity * dt;
            rigidVelocity *= damping;
            rigidPosition += rigidVelocity * dt;

            if (glm::length(rigidAngularVelocity) > 0.0f) {
                float angle = glm::length(rigidAngularVelocity) * dt;
                glm::vec3 axis = glm::normalize(rigidAngularVelocity);
                glm::quat deltaRotation = glm::angleAxis(angle, axis);
                rigidRotation = deltaRotation * rigidRotation;
                rigidAngularVelocity *= damping;
            }

            updateRigidTransform();

            float minY = std::numeric_limits<float>::max();
            for (size_t i = 0; i < numParticles; i++) {
                minY = std::min(minY, positions[i * 3 + 1]);
            }

        }
        return;
    }

    for (size_t i = 0; i < numParticles; i++) {
        if (invMasses[i] == 0.0f) continue;

        VectorMath::vecAdd(velocities, i, {gravity.x, gravity.y, gravity.z}, 0, dt);
        VectorMath::vecScale(velocities, i, damping);
        VectorMath::vecCopy(prevPositions, i, positions, i);
        VectorMath::vecAdd(positions, i, velocities, i, dt);

        if (positions[3 * i + 1] < -5.0f) {
            VectorMath::vecCopy(positions, i, prevPositions, i);
            positions[3 * i + 1] = -5.0f;
            velocities[3 * i + 1] = -5.0f;
        }
    }
}

void SoftBody::solve(float dt) {
    if (rigidMode) {
        return;
    }
    for (int i = 0; i < 5; i++) {
        solveEdges(edgeCompliance, dt);
        for (int j = 0; j < 2; j++) {
            solveVolumes(volCompliance, dt);
        }
    }
}

float SoftBody::getTetVolume(int nr) {
    int id0 = tetIds[4 * nr];
    int id1 = tetIds[4 * nr + 1];
    int id2 = tetIds[4 * nr + 2];
    int id3 = tetIds[4 * nr + 3];

    VectorMath::vecSetDiff(tempBuffer, 0, positions, id1, positions, id0);
    VectorMath::vecSetDiff(tempBuffer, 1, positions, id2, positions, id0);
    VectorMath::vecSetDiff(tempBuffer, 2, positions, id3, positions, id0);
    VectorMath::vecSetCross(tempBuffer, 3, tempBuffer, 0, tempBuffer, 1);

    return VectorMath::vecDot(tempBuffer, 3, tempBuffer, 2) / 6.0f;
}

void SoftBody::initPhysics() {
    std::fill(invMasses.begin(), invMasses.end(), 0.0f);
    std::fill(restVols.begin(), restVols.end(), 0.0f);
    std::fill(edgeLambdas.begin(), edgeLambdas.end(), 0.0f);
    std::fill(volLambdas.begin(), volLambdas.end(), 0.0f);

    float cusum_vol = 1.0;
    for (size_t i = 0; i < numTets; i++) {
        float vol = getTetVolume(i);
        restVols[i] = vol;
        float pInvMass = vol > 0.0f ? 1.0f / (vol / 4000000.0f) : 1000000.0f;

        for (int j = 0; j < 4; j++) {
            invMasses[tetIds[4 * i + j]] += pInvMass;
        }
    }

    for (size_t i = 0; i < edgeLengths.size(); i++) {
        int id0 = edgeIds[2 * i];
        int id1 = edgeIds[2 * i + 1];
        edgeLengths[i] = std::sqrt(VectorMath::vecDistSquared(positions, id0, positions, id1));
    }
}

void SoftBody::solveEdges(float compliance, float dt) {
    float alpha = compliance / (dt * dt);

    for (size_t i = 0; i < edgeLengths.size(); i++) {
        int id0 = edgeIds[2 * i];
        int id1 = edgeIds[2 * i + 1];
        float w0 = invMasses[id0];
        float w1 = invMasses[id1];
        float w = w0 + w1;
        if (w == 0.0f) continue;

        VectorMath::vecSetDiff(grads, 0, positions, id0, positions, id1);
        float len = std::sqrt(VectorMath::vecLengthSquared(grads, 0));
        if (len == 0.0f) continue;

        VectorMath::vecScale(grads, 0, 1.0f / len);
        float restLen = edgeLengths[i];
        float C = len - restLen;

        float dLambda = -(C + alpha * edgeLambdas[i]) / (w + alpha);
        edgeLambdas[i] += dLambda;

        VectorMath::vecAdd(positions, id0, grads, 0, dLambda * w0);
        VectorMath::vecAdd(positions, id1, grads, 0, -dLambda * w1);
    }
}

void SoftBody::solveVolumes(float compliance, float dt) {
    float alpha = compliance / (dt * dt);

    for (size_t i = 0; i < numTets; i++) {
        float w = 0.0f;

        for (int j = 0; j < 4; j++) {
            int id0 = tetIds[4 * i + volIdOrder[j][0]];
            int id1 = tetIds[4 * i + volIdOrder[j][1]];
            int id2 = tetIds[4 * i + volIdOrder[j][2]];

            VectorMath::vecSetDiff(tempBuffer, 0, positions, id1, positions, id0);
            VectorMath::vecSetDiff(tempBuffer, 1, positions, id2, positions, id0);
            VectorMath::vecSetCross(grads, j, tempBuffer, 0, tempBuffer, 1);
            VectorMath::vecScale(grads, j, 1.0f/6.0f);

            w += invMasses[tetIds[4 * i + j]] * VectorMath::vecLengthSquared(grads, j);
        }

        if (w == 0.0f) continue;

        float vol = getTetVolume(i);
        float restVol = restVols[i];
        float C = vol - restVol;

        float dLambda = -(C + alpha * volLambdas[i]) / (w + alpha);
        volLambdas[i] += dLambda;

        for (int j = 0; j < 4; j++) {
            int id = tetIds[4 * i + j];
            VectorMath::vecAdd(positions, id, grads, j, dLambda * invMasses[id]);
        }
    }
}

void SoftBody::postSolve(float dt) {
    if (rigidMode) {
        return;
    }
    for (size_t i = 0; i < numParticles; i++) {
        if (invMasses[i] == 0.0f) continue;
        VectorMath::vecSetDiff(velocities, i, positions, i, prevPositions, i, 1.0f / dt);
    }
}

void SoftBody::setupTetMesh() {
    if (tetVAO != 0) {
        glDeleteVertexArrays(1, &tetVAO);
        tetVAO = 0;
    }
    if (tetVBO != 0) {
        glDeleteBuffers(1, &tetVBO);
        tetVBO = 0;
    }
    if (tetEBO != 0) {
        glDeleteBuffers(1, &tetEBO);
        tetEBO = 0;
    }

    glGenVertexArrays(1, &tetVAO);
    glGenBuffers(1, &tetVBO);
    glGenBuffers(1, &tetEBO);

    glBindVertexArray(tetVAO);

    std::vector<float> edgeVertices;
    for (size_t i = 0; i < meshData.tetEdgeIds.size(); i += 2) {
        int id0 = meshData.tetEdgeIds[i];
        int id1 = meshData.tetEdgeIds[i + 1];

        edgeVertices.push_back(positions[id0 * 3]);
        edgeVertices.push_back(positions[id0 * 3 + 1]);
        edgeVertices.push_back(positions[id0 * 3 + 2]);

        edgeVertices.push_back(positions[id1 * 3]);
        edgeVertices.push_back(positions[id1 * 3 + 1]);
        edgeVertices.push_back(positions[id1 * 3 + 2]);
    }

    glBindBuffer(GL_ARRAY_BUFFER, tetVBO);
    glBufferData(GL_ARRAY_BUFFER, edgeVertices.size() * sizeof(float),
                 edgeVertices.data(), GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SoftBody::updateTetMeshes() {
    std::vector<float> edgeVertices;
    for (size_t i = 0; i < meshData.tetEdgeIds.size(); i += 2) {
        int id0 = meshData.tetEdgeIds[i];
        int id1 = meshData.tetEdgeIds[i + 1];

        edgeVertices.push_back(positions[id0 * 3]);
        edgeVertices.push_back(positions[id0 * 3 + 1]);
        edgeVertices.push_back(positions[id0 * 3 + 2]);

        edgeVertices.push_back(positions[id1 * 3]);
        edgeVertices.push_back(positions[id1 * 3 + 1]);
        edgeVertices.push_back(positions[id1 * 3 + 2]);
    }

    glBindBuffer(GL_ARRAY_BUFFER, tetVBO);
    glBufferData(GL_ARRAY_BUFFER, edgeVertices.size() * sizeof(float),
                 edgeVertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SoftBody::drawTetMesh(ShaderProgram& shader) {
    if (!showTetMesh) return;

    shader.use();
    glBindVertexArray(tetVAO);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawArrays(GL_LINES, 0, meshData.tetEdgeIds.size());
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindVertexArray(0);
}

void SoftBody::startGrab(const glm::vec3& pos) {
    if (rigidMode) {
        grabOffset = pos - rigidPosition;
        isDragging = true;
        return;
    }

    float minD2 = std::numeric_limits<float>::max();
    grabId = -1;

    struct ParticleDistance {
        int id;
        float distance;
    };

    std::vector<ParticleDistance> sortedParticles;
    activeParticles.clear();

    for (size_t i = 0; i < numParticles; i++) {
        glm::vec3 particlePos(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        glm::vec3 diff = particlePos - pos;
        float d2 = glm::dot(diff, diff);
        if (d2 < minD2) {
            minD2 = d2;
            grabId = i;
        }
        sortedParticles.push_back({(int)i, d2});
    }

    std::sort(sortedParticles.begin(), sortedParticles.end(),
              [](const ParticleDistance& a, const ParticleDistance& b) {
                  return a.distance < b.distance;
              });

    int numToSelect = static_cast<int>(sortedParticles.size() * 1.0);
    numToSelect = std::max(1, numToSelect);
    std::cout << "numToSelect: " << numToSelect  << std::endl;
    std::cout << "sortedParticles.size(): " << sortedParticles.size()  << std::endl;

    for (int i = 0; i < numToSelect; i++) {
        activeParticles.push_back(sortedParticles[i].id);
    }

    oldInvMasses = invMasses;
    std::fill(invMasses.begin(), invMasses.end(), 0.0f);

    for (int id : activeParticles) {
        invMasses[id] = oldInvMasses[id];
    }

    invMasses[grabId] = 0.0f;

    int last = numParticles -1;

    positions[grabId * 3] = pos.x;
    positions[grabId * 3 + 1] = pos.y;
    positions[grabId * 3 + 2] = pos.z;
}

void SoftBody::moveGrabbed(const glm::vec3& pos, const glm::vec3& vel) {
    if (rigidMode) {
        rigidPosition = pos - grabOffset;
        updateRigidTransform();
        return;
    }

    if (grabId >= 0) {
        positions[grabId * 3] = pos.x;
        positions[grabId * 3 + 1] = pos.y;
        positions[grabId * 3 + 2] = pos.z;
    }
}

void SoftBody::endGrab(const glm::vec3& pos, const glm::vec3& vel) {
    if (rigidMode) {
        isDragging = false;
        rigidVelocity = glm::vec3(0.0f);
        rigidAngularVelocity = glm::vec3(0.0f);
        return;
    }

    if (grabId >= 0) {
        for (int id : activeParticles) {
            invMasses[id] = oldInvMasses[id];
        }

        velocities[grabId * 3] = vel.x;
        velocities[grabId * 3 + 1] = vel.y;
        velocities[grabId * 3 + 2] = vel.z;

        grabId = -1;
        activeParticles.clear();
    }
}

void SoftBody::applyShapeRestoration(float strength) {
    for (size_t i = 0; i < numParticles; i++) {
        glm::vec3 restPos(meshData.verts[i * 3], meshData.verts[i * 3 + 1], meshData.verts[i * 3 + 2]);
        glm::vec3 currentPos(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);

        glm::vec3 correction = (restPos - currentPos) * strength;

        positions[i * 3] += correction.x;
        positions[i * 3 + 1] += correction.y;
        positions[i * 3 + 2] += correction.z;
    }

    initPhysics();
}

SoftBody::MeshData SoftBody::loadTetMesh(const std::string& filename) {
    SoftBody::MeshData meshData;
    std::ifstream file(filename);
    std::string line;
    bool readingVertices = false, readingTetrahedra = false, readingEdges = false, readingSurfaceTris = false;

    while (std::getline(file, line)) {
        if (line == "VERTICES") {
            readingVertices = true; readingTetrahedra = false; readingEdges = false; readingSurfaceTris = false;
            continue;
        }
        if (line == "TETRAHEDRA") {
            readingVertices = false; readingTetrahedra = true; readingEdges = false; readingSurfaceTris = false;
            continue;
        }
        if (line == "EDGES") {
            readingVertices = false; readingTetrahedra = false; readingEdges = true; readingSurfaceTris = false;
            continue;
        }
        if (line == "SURFACE_TRIANGLES") {
            readingVertices = false; readingTetrahedra = false; readingEdges = false; readingSurfaceTris = true;
            continue;
        }

        std::istringstream ss(line);
        if (readingVertices) {
            float x, y, z;
            ss >> x >> y >> z;
            meshData.verts.push_back(x);
            meshData.verts.push_back(y);
            meshData.verts.push_back(z);
        }
        else if (readingTetrahedra) {
            int v0, v1, v2, v3;
            ss >> v0 >> v1 >> v2 >> v3;
            meshData.tetIds.push_back(v0);
            meshData.tetIds.push_back(v1);
            meshData.tetIds.push_back(v2);
            meshData.tetIds.push_back(v3);
        }
        else if (readingEdges) {
            int e0, e1;
            ss >> e0 >> e1;
            meshData.tetEdgeIds.push_back(e0);
            meshData.tetEdgeIds.push_back(e1);
        }
        else if (readingSurfaceTris) {
            int t0, t1, t2;
            ss >> t0 >> t1 >> t2;
            meshData.tetSurfaceTriIds.push_back(t0);
            meshData.tetSurfaceTriIds.push_back(t1);
            meshData.tetSurfaceTriIds.push_back(t2);
        }
    }
    return meshData;
}

SoftBody::~SoftBody() {
    deleteBuffers();
}

void SoftBody::deleteBuffers() {
    if (tetVAO != 0) {
        glDeleteVertexArrays(1, &tetVAO);
        tetVAO = 0;
    }
    if (tetVBO != 0) {
        glDeleteBuffers(1, &tetVBO);
        tetVBO = 0;
    }
    if (tetEBO != 0) {
        glDeleteBuffers(1, &tetEBO);
        tetEBO = 0;
    }

    for (size_t i = 0; i < visVAOs.size(); i++) {
        if (visVAOs[i] != 0) {
            glDeleteVertexArrays(1, &visVAOs[i]);
            glDeleteBuffers(1, &visVBOs[i]);
            glDeleteBuffers(1, &visEBOs[i]);
            glDeleteBuffers(1, &visNormalVBOs[i]);
        }
    }
    visVAOs.clear();
    visVBOs.clear();
    visEBOs.clear();
    visNormalVBOs.clear();
}

SoftBody::SoftBody(const MeshData& tetMesh,
                   const std::vector<MeshData>& visMeshes,
                   float edgeCompliance,
                   float volCompliance)
    : meshData(tetMesh)
    , vismeshDataArray(visMeshes)
    , edgeCompliance(edgeCompliance)
    , volCompliance(volCompliance)
    , grabId(-1)
    , grabInvMass(0.0f)
    , damping(0.99f)
{
    std::cout << "=== SoftBody Constructor Debug ===" << std::endl;

    std::cout << "TetMesh Info:" << std::endl;
    std::cout << "  Vertices: " << tetMesh.verts.size() / 3 << std::endl;
    std::cout << "  Tetrahedra: " << tetMesh.tetIds.size() / 4 << std::endl;
    std::cout << "  Edges: " << tetMesh.tetEdgeIds.size() / 2 << std::endl;
    std::cout << "  Surface Triangles: " << tetMesh.tetSurfaceTriIds.size() / 3 << std::endl;

    std::cout << "VisMesh Count: " << visMeshes.size() << std::endl;
    for (size_t i = 0; i < visMeshes.size(); i++) {
        std::cout << "VisMesh " << i << " Info:" << std::endl;
        std::cout << "  Vertices: " << visMeshes[i].verts.size() / 3 << std::endl;
        std::cout << "  Surface Triangles: " << visMeshes[i].tetSurfaceTriIds.size() / 3 << std::endl;
    }

    numParticles = tetMesh.verts.size() / 3;
    numTets = tetMesh.tetIds.size() / 4;

    vis_positions_array.resize(visMeshes.size());
    visSurfaceTriIds_array.resize(visMeshes.size());
    skinningInfo_array.resize(visMeshes.size());

    numVisVerts = 0;
    numVisParticles = 0;

    original_vis_positions_array.resize(visMeshes.size());

    for (size_t i = 0; i < visMeshes.size(); i++) {
        vis_positions_array[i] = visMeshes[i].verts;
        visSurfaceTriIds_array[i] = visMeshes[i].tetSurfaceTriIds;

        size_t meshVisVerts = visMeshes[i].verts.size() / 3;
        numVisVerts += meshVisVerts;
        numVisParticles += meshVisVerts;

        skinningInfo_array[i].resize(4 * meshVisVerts, -1.0f);
        original_vis_positions_array[i] = visMeshes[i].verts;
    }

    visVAOs.resize(visMeshes.size(), 0);
    visVBOs.resize(visMeshes.size(), 0);
    visEBOs.resize(visMeshes.size(), 0);
    visNormalVBOs.resize(visMeshes.size(), 0);

    positions = tetMesh.verts;
    prevPositions = tetMesh.verts;
    velocities.resize(3 * numParticles, 0.0f);

    tetIds = tetMesh.tetIds;
    tetSurfaceTriIds = tetMesh.tetSurfaceTriIds;
    edgeIds = tetMesh.tetEdgeIds;

    restVols.resize(numTets, 0.0f);
    edgeLengths.resize(edgeIds.size() / 2, 0.0f);
    invMasses.resize(numParticles, 0.0f);

    tempBuffer.resize(4 * 3, 0.0f);
    grads.resize(4 * 3, 0.0f);

    edgeLambdas.resize(edgeIds.size() / 2, 0.0f);
    volLambdas.resize(numTets, 0.0f);

    for (size_t i = 0; i < visMeshes.size(); i++) {
        computeSkinningInfoForMesh(visMeshes[i].verts, skinningInfo_array[i]);
    }

    initPhysics();
    setupVisMeshes();
    setupTetMesh();

    showTetMesh = true;
    modelMatrix = glm::mat4(1.0f);
    std::cout << "=== Constructor Complete ===" << std::endl;

    originalPositions = tetMesh.verts;

    rigidMode = false;
    isDragging = false;
    rigidRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    rigidVelocity = glm::vec3(0.0f);
    rigidAngularVelocity = glm::vec3(0.0f);
}

void SoftBody::setupVisMeshes() {
    for (size_t i = 0; i < visVAOs.size(); i++) {
        if (visVAOs[i] != 0) {
            glDeleteVertexArrays(1, &visVAOs[i]);
            glDeleteBuffers(1, &visVBOs[i]);
            glDeleteBuffers(1, &visEBOs[i]);
            glDeleteBuffers(1, &visNormalVBOs[i]);
        }
    }

    for (size_t i = 0; i < vismeshDataArray.size(); i++) {
        glGenVertexArrays(1, &visVAOs[i]);
        glGenBuffers(1, &visVBOs[i]);
        glGenBuffers(1, &visEBOs[i]);
        glGenBuffers(1, &visNormalVBOs[i]);

        glBindVertexArray(visVAOs[i]);

        glBindBuffer(GL_ARRAY_BUFFER, visVBOs[i]);
        glBufferData(GL_ARRAY_BUFFER, vis_positions_array[i].size() * sizeof(float),
                     vis_positions_array[i].data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, visNormalVBOs[i]);
        computeVisNormalsForMesh(i);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, visEBOs[i]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     visSurfaceTriIds_array[i].size() * sizeof(int),
                     visSurfaceTriIds_array[i].data(), GL_STATIC_DRAW);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SoftBody::updateVisMeshes() {
    for (size_t meshIdx = 0; meshIdx < vismeshDataArray.size(); meshIdx++) {
        size_t numMeshVerts = vis_positions_array[meshIdx].size() / 3;

        int nr = 0;
        for (size_t i = 0; i < numMeshVerts; i++) {
            int tetNr = static_cast<int>(skinningInfo_array[meshIdx][nr++]) * 4;
            if (tetNr < 0) {
                nr += 3;
                continue;
            }
            if (tetNr + 3 >= (int)tetIds.size()) {
                static bool warned = false;
                if (!warned) {
                    printf("[WARN] updateVisMeshes mesh %zu: tetNr=%d out of bounds (tetIds.size=%zu, vert %zu)\n",
                           meshIdx, tetNr, tetIds.size(), i);
                    warned = true;
                }
                nr += 3;
                continue;
            }

            float b0 = skinningInfo_array[meshIdx][nr++];
            float b1 = skinningInfo_array[meshIdx][nr++];
            float b2 = skinningInfo_array[meshIdx][nr++];
            float b3 = 1.0f - b0 - b1 - b2;

            int id0 = tetIds[tetNr++];
            int id1 = tetIds[tetNr++];
            int id2 = tetIds[tetNr++];
            int id3 = tetIds[tetNr++];

            int maxPosIdx = (int)(positions.size() / 3);
            if (id0 < 0 || id0 >= maxPosIdx || id1 < 0 || id1 >= maxPosIdx ||
                id2 < 0 || id2 >= maxPosIdx || id3 < 0 || id3 >= maxPosIdx) {
                static bool warned2 = false;
                if (!warned2) {
                    printf("[WARN] updateVisMeshes: vertex index out of bounds! "
                           "ids=%d,%d,%d,%d maxIdx=%d\n", id0, id1, id2, id3, maxPosIdx);
                    warned2 = true;
                }
                continue;
            }

            VectorMath::vecSetZero(vis_positions_array[meshIdx], i);
            VectorMath::vecAdd(vis_positions_array[meshIdx], i, positions, id0, b0);
            VectorMath::vecAdd(vis_positions_array[meshIdx], i, positions, id1, b1);
            VectorMath::vecAdd(vis_positions_array[meshIdx], i, positions, id2, b2);
            VectorMath::vecAdd(vis_positions_array[meshIdx], i, positions, id3, b3);
        }

        glBindBuffer(GL_ARRAY_BUFFER, visVBOs[meshIdx]);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        vis_positions_array[meshIdx].size() * sizeof(float),
                        vis_positions_array[meshIdx].data());

        computeVisNormalsForMesh(meshIdx);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SoftBody::computeVisNormalsForMesh(size_t meshIdx) {
    size_t numMeshVerts = vis_positions_array[meshIdx].size() / 3;
    std::vector<glm::vec3> normals(numMeshVerts, glm::vec3(0.0f));

    for (size_t i = 0; i < visSurfaceTriIds_array[meshIdx].size(); i += 3) {
        int id0 = visSurfaceTriIds_array[meshIdx][i];
        int id1 = visSurfaceTriIds_array[meshIdx][i + 1];
        int id2 = visSurfaceTriIds_array[meshIdx][i + 2];

        if (id0 < 0 || id0 >= (int)numMeshVerts ||
            id1 < 0 || id1 >= (int)numMeshVerts ||
            id2 < 0 || id2 >= (int)numMeshVerts) {
            static bool warned = false;
            if (!warned) {
                printf("[WARN] computeVisNormals mesh %zu: INDEX OUT OF BOUNDS! "
                       "id0=%d id1=%d id2=%d numVerts=%zu (tri %zu)\n",
                       meshIdx, id0, id1, id2, numMeshVerts, i/3);
                warned = true;
            }
            continue;
        }

        glm::vec3 p0(vis_positions_array[meshIdx][id0 * 3],
                     vis_positions_array[meshIdx][id0 * 3 + 1],
                     vis_positions_array[meshIdx][id0 * 3 + 2]);
        glm::vec3 p1(vis_positions_array[meshIdx][id1 * 3],
                     vis_positions_array[meshIdx][id1 * 3 + 1],
                     vis_positions_array[meshIdx][id1 * 3 + 2]);
        glm::vec3 p2(vis_positions_array[meshIdx][id2 * 3],
                     vis_positions_array[meshIdx][id2 * 3 + 1],
                     vis_positions_array[meshIdx][id2 * 3 + 2]);

        glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));

        normals[id0] += normal;
        normals[id1] += normal;
        normals[id2] += normal;
    }

    std::vector<float> normalBuffer;
    normalBuffer.reserve(numMeshVerts * 3);

    for (const auto& n : normals) {
        glm::vec3 normalized = glm::length(n) > 0.0f ? glm::normalize(n) : n;
        normalBuffer.push_back(normalized.x);
        normalBuffer.push_back(normalized.y);
        normalBuffer.push_back(normalized.z);
    }

    glBindBuffer(GL_ARRAY_BUFFER, visNormalVBOs[meshIdx]);
    glBufferData(GL_ARRAY_BUFFER, normalBuffer.size() * sizeof(float),
                 normalBuffer.data(), GL_DYNAMIC_DRAW);
}

void SoftBody::computeSkinningInfoForMesh(const std::vector<float>& visVerts,
                                          std::vector<float>& skinningInfoOut) {
    size_t numMeshVerts = visVerts.size() / 3;

    std::cout << "Computing skinning for mesh with " << numMeshVerts << " vertices" << std::endl;

    glm::vec3 tetMin(std::numeric_limits<float>::max());
    glm::vec3 tetMax(std::numeric_limits<float>::lowest());
    glm::vec3 visMin(std::numeric_limits<float>::max());
    glm::vec3 visMax(std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < positions.size(); i += 3) {
        tetMin.x = std::min(tetMin.x, positions[i]);
        tetMin.y = std::min(tetMin.y, positions[i + 1]);
        tetMin.z = std::min(tetMin.z, positions[i + 2]);
        tetMax.x = std::max(tetMax.x, positions[i]);
        tetMax.y = std::max(tetMax.y, positions[i + 1]);
        tetMax.z = std::max(tetMax.z, positions[i + 2]);
    }

    for (size_t i = 0; i < visVerts.size(); i += 3) {
        visMin.x = std::min(visMin.x, visVerts[i]);
        visMin.y = std::min(visMin.y, visVerts[i + 1]);
        visMin.z = std::min(visMin.z, visVerts[i + 2]);
        visMax.x = std::max(visMax.x, visVerts[i]);
        visMax.y = std::max(visMax.y, visVerts[i + 1]);
        visMax.z = std::max(visMax.z, visVerts[i + 2]);
    }

    glm::vec3 tetSize = tetMax - tetMin;
    glm::vec3 visSize = visMax - visMin;
    float maxSize = std::max({tetSize.x, tetSize.y, tetSize.z});
    float spacing = maxSize * 1.0f;

    Hash hash(spacing, numMeshVerts);
    hash.create(visVerts);

    skinningInfoOut.assign(4 * numMeshVerts, -1.0f);
    std::vector<float> minDist(numMeshVerts, std::numeric_limits<float>::max());
    const float border = 0.05f;

    std::vector<float> tetCenter(3, 0.0f);
    std::vector<float> mat(9, 0.0f);
    std::vector<float> bary(4, 0.0f);

    for (size_t i = 0; i < numTets; i++) {
        std::fill(tetCenter.begin(), tetCenter.end(), 0.0f);
        for (int j = 0; j < 4; j++) {
            VectorMath::vecAdd(tetCenter, 0, positions, tetIds[4 * i + j], 0.25f);
        }

        float rMax = 0.0f;
        for (int j = 0; j < 4; j++) {
            float r2 = VectorMath::vecDistSquared(tetCenter, 0, positions, tetIds[4 * i + j]);
            rMax = std::max(rMax, std::sqrt(r2));
        }
        rMax += border;

        hash.query(tetCenter, 0, rMax);
        if (hash.querySize == 0) continue;

        int id0 = tetIds[4 * i];
        int id1 = tetIds[4 * i + 1];
        int id2 = tetIds[4 * i + 2];
        int id3 = tetIds[4 * i + 3];

        VectorMath::vecSetDiff(mat, 0, positions, id0, positions, id3);
        VectorMath::vecSetDiff(mat, 1, positions, id1, positions, id3);
        VectorMath::vecSetDiff(mat, 2, positions, id2, positions, id3);
        VectorMath::matSetInverse(mat);

        for (int j = 0; j < hash.querySize; j++) {
            int id = hash.queryIds[j];

            if (minDist[id] <= 0.0f) continue;
            if (VectorMath::vecDistSquared(visVerts, id, tetCenter, 0) > rMax * rMax) continue;

            VectorMath::vecSetDiff(bary, 0, visVerts, id, positions, id3);
            VectorMath::matSetMult(mat, bary, 0, bary, 0);
            bary[3] = 1.0f - bary[0] - bary[1] - bary[2];

            float dist = 0.0f;
            for (int k = 0; k < 4; k++) {
                dist = std::max(dist, -bary[k]);
            }

            if (dist < minDist[id]) {
                minDist[id] = dist;
                skinningInfoOut[4 * id] = static_cast<float>(i);
                skinningInfoOut[4 * id + 1] = bary[0];
                skinningInfoOut[4 * id + 2] = bary[1];
                skinningInfoOut[4 * id + 3] = bary[2];
            }
        }
    }
}

void SoftBody::draw_AllVisMeshes(ShaderProgram& shader, glm::vec3 camPos,
                                 std::vector<glm::vec4>& meshColors) {
    shader.use();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glm::vec3 cameraPos = camPos;

    struct GlobalTriangleInfo {
        size_t meshIndex;
        size_t triangleIndex;
        float distance;
    };

    std::vector<GlobalTriangleInfo> allTriangles;

    for (size_t meshIdx = 0; meshIdx < visVAOs.size(); meshIdx++) {
        const auto& positions = vis_positions_array[meshIdx];
        const auto& indices = visSurfaceTriIds_array[meshIdx];

        for (size_t j = 0; j < indices.size(); j += 3) {
            int idx1 = indices[j];
            int idx2 = indices[j + 1];
            int idx3 = indices[j + 2];

            glm::vec3 v1(positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
            glm::vec3 v2(positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);
            glm::vec3 v3(positions[idx3 * 3], positions[idx3 * 3 + 1], positions[idx3 * 3 + 2]);

            glm::vec3 center = (v1 + v2 + v3) / 3.0f;
            float distance = glm::length(cameraPos - center);

            allTriangles.push_back({meshIdx, j / 3, distance});
        }
    }

    std::sort(allTriangles.begin(), allTriangles.end(),
              [](const GlobalTriangleInfo& a, const GlobalTriangleInfo& b) {
                  return a.distance > b.distance;
              });

    GLuint lastVAO = -1;
    glm::vec4 lastColor;
    bool colorInitialized = false;

    for (const auto& tri : allTriangles) {
        size_t meshIdx = tri.meshIndex;
        size_t triIdx = tri.triangleIndex;

        if (lastVAO != visVAOs[meshIdx]) {
            glBindVertexArray(visVAOs[meshIdx]);
            lastVAO = visVAOs[meshIdx];
        }

        glm::vec4 currentColor = meshColors[meshIdx % meshColors.size()];
        if (!colorInitialized || lastColor != currentColor) {
            shader.setUniform("vertColor", currentColor);
            lastColor = currentColor;
            colorInitialized = true;
        }

        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT,
                       (void*)(triIdx * 3 * sizeof(GLuint)));
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glBindVertexArray(0);
}

void SoftBody::HandleGroup::updateCenterPosition(const std::vector<float>& positions) {
    if (vertices.empty()) return;

    glm::vec3 sum(0.0f);
    for (int idx : vertices) {
        sum.x += positions[idx * 3];
        sum.y += positions[idx * 3 + 1];
        sum.z += positions[idx * 3 + 2];
    }
    centerPosition = sum / static_cast<float>(vertices.size());
}

void SoftBody::HandleGroup::storeRelativePositions(const std::vector<float>& positions) {
    relativePositions.clear();
    updateCenterPosition(positions);

    for (int idx : vertices) {
        glm::vec3 vertPos(positions[idx * 3],
                          positions[idx * 3 + 1],
                          positions[idx * 3 + 2]);
        relativePositions.push_back(vertPos - centerPosition);
    }
}

int SoftBody::findClosestVertex(const glm::vec3& position) {
    float minD2 = std::numeric_limits<float>::max();
    int closestId = -1;

    for (size_t i = 0; i < numParticles; i++) {
        glm::vec3 particlePos(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        glm::vec3 diff = particlePos - position;
        float d2 = glm::dot(diff, diff);

        if (d2 < minD2) {
            minD2 = d2;
            closestId = i;
        }
    }

    return closestId;
}

glm::vec3 SoftBody::getVertexPosition(int index) const {
    if (index >= 0 && index < numParticles) {
        return glm::vec3(positions[index * 3],
                         positions[index * 3 + 1],
                         positions[index * 3 + 2]);
    }
    return glm::vec3(0.0f);
}

void SoftBody::addHandleVertex(const glm::vec3& position) {
    int vertexId = findClosestVertex(position);
    if (vertexId >= 0) {
        handleVertices.push_back(vertexId);
        invMasses[vertexId] = 0.0f;
    }
}

void SoftBody::addHandleVertexByIndex(int vertexId) {
    if (vertexId >= 0 && vertexId < numParticles) {
        handleVertices.push_back(vertexId);
        invMasses[vertexId] = 0.0f;
    }
}

void SoftBody::clearHandles() {
    handleVertices.clear();
    initPhysics();
}

bool SoftBody::createHandleGroup(const glm::vec3& position, float radius) {
    if (handleGroups.size() >= MAX_HANDLE_GROUPS) {
        return false;
    }

    HandleGroup group;
    group.radius = radius;

    float minDist = std::numeric_limits<float>::max();
    int centerIdx = -1;

    for (size_t i = 0; i < numParticles; i++) {
        glm::vec3 vertPos(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        float dist = glm::length(position - vertPos);

        if (dist < minDist) {
            minDist = dist;
            centerIdx = i;
        }
    }

    if (centerIdx < 0) return false;

    group.centerVertex = centerIdx;

    for (size_t i = 0; i < numParticles; i++) {
        glm::vec3 vertPos(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        glm::vec3 centerPos(positions[centerIdx * 3],
                            positions[centerIdx * 3 + 1],
                            positions[centerIdx * 3 + 2]);

        float dist = glm::length(vertPos - centerPos);
        if (dist <= radius) {
            group.vertices.push_back(i);

            velocities[i * 3] = 0.0f;
            velocities[i * 3 + 1] = 0.0f;
            velocities[i * 3 + 2] = 0.0f;

            prevPositions[i * 3] = positions[i * 3];
            prevPositions[i * 3 + 1] = positions[i * 3 + 1];
            prevPositions[i * 3 + 2] = positions[i * 3 + 2];

            invMasses[i] = 0.0f;
        }
    }

    group.storeRelativePositions(positions);
    handleGroups.push_back(group);

    return true;
}

void SoftBody::clearHandleGroups() {
    for (const auto& group : handleGroups) {
        for (int idx : group.vertices) {
            if (idx >= 0 && idx < numParticles) {
                velocities[idx * 3] = 0.0f;
                velocities[idx * 3 + 1] = 0.0f;
                velocities[idx * 3 + 2] = 0.0f;
            }
        }
    }

    currentGrabMode = GRAB_NONE;
    activeHandleGroup = -1;
    handleGroups.clear();

    initPhysics();

    std::cout << "Handle groups cleared" << std::endl;
}

void SoftBody::reapplyHandleConstraints() {
    for (const auto& group : handleGroups) {
        for (int idx : group.vertices) {
            if (idx >= 0 && idx < (int)numParticles) {
                invMasses[idx] = 0.0f;
            }
        }
    }
    for (int idx : handleVertices) {
        if (idx >= 0 && idx < (int)numParticles) {
            invMasses[idx] = 0.0f;
        }
    }
}

int SoftBody::findHandleGroupAtPosition(const glm::vec3& position, float threshold) {
    for (size_t g = 0; g < handleGroups.size(); g++) {
        handleGroups[g].updateCenterPosition(positions);
        float dist = glm::length(position - handleGroups[g].centerPosition);
        if (dist <= threshold) {
            return static_cast<int>(g);
        }
    }
    return -1;
}

std::vector<glm::vec3> SoftBody::getHandleGroupPositions(int groupIndex) const {
    std::vector<glm::vec3> positions_out;
    if (groupIndex >= 0 && groupIndex < handleGroups.size()) {
        for (int idx : handleGroups[groupIndex].vertices) {
            positions_out.push_back(glm::vec3(
                positions[idx * 3],
                positions[idx * 3 + 1],
                positions[idx * 3 + 2]
                ));
        }
    }
    return positions_out;
}

void SoftBody::startGrabHandleGroup(const glm::vec3& hitPosition) {
    activeHandleGroup = -1;
    float minDist = std::numeric_limits<float>::max();

    for (size_t g = 0; g < handleGroups.size(); g++) {
        for (int vertIdx : handleGroups[g].vertices) {
            glm::vec3 vertPos(positions[vertIdx * 3],
                              positions[vertIdx * 3 + 1],
                              positions[vertIdx * 3 + 2]);
            float dist = glm::length(hitPosition - vertPos);

            if (dist < minDist) {
                minDist = dist;
                activeHandleGroup = g;
            }
        }
    }

    if (activeHandleGroup >= 0) {
        handleGroups[activeHandleGroup].updateCenterPosition(positions);
        grabOffset = hitPosition - handleGroups[activeHandleGroup].centerPosition;

        for (int idx : handleGroups[activeHandleGroup].vertices) {
            invMasses[idx] = 0.0f;
        }

        std::cout << "Grabbed handle group " << activeHandleGroup
                  << " with offset: " << grabOffset.x << ", "
                  << grabOffset.y << ", " << grabOffset.z << std::endl;
    }
}

bool SoftBody::tryStartGrabHandleGroup(const glm::vec3& hitPosition, float threshold) {
    int groupIndex = findHandleGroupAtPosition(hitPosition, threshold);

    if (groupIndex >= 0) {
        activeHandleGroup = groupIndex;
        handleGroups[activeHandleGroup].updateCenterPosition(positions);
        grabOffset = hitPosition - handleGroups[activeHandleGroup].centerPosition;

        std::cout << "Grabbed handle group " << activeHandleGroup
                  << " with offset: " << grabOffset.x << ", "
                  << grabOffset.y << ", " << grabOffset.z << std::endl;

        return true;
    }

    return false;
}

void SoftBody::moveGrabbedHandleGroup(const glm::vec3& newPosition, const glm::vec3& velocity) {
    if (activeHandleGroup < 0 || activeHandleGroup >= handleGroups.size()) return;

    HandleGroup& group = handleGroups[activeHandleGroup];
    glm::vec3 newCenterPos = newPosition - grabOffset;

    for (size_t i = 0; i < group.vertices.size(); i++) {
        int idx = group.vertices[i];
        glm::vec3 newVertPos = newCenterPos + group.relativePositions[i];

        positions[idx * 3] = newVertPos.x;
        positions[idx * 3 + 1] = newVertPos.y;
        positions[idx * 3 + 2] = newVertPos.z;

        velocities[idx * 3] = velocity.x;
        velocities[idx * 3 + 1] = velocity.y;
        velocities[idx * 3 + 2] = velocity.z;
    }

    group.centerPosition = newCenterPos;
}

void SoftBody::endGrabHandleGroup(const glm::vec3& position, const glm::vec3& velocity) {
    if (activeHandleGroup >= 0) {
        for (int idx : handleGroups[activeHandleGroup].vertices) {
            invMasses[idx] = 0.0f;
        }

        std::cout << "Released handle group " << activeHandleGroup << std::endl;
        activeHandleGroup = -1;
    }
}

void SoftBody::smartGrab(const glm::vec3& hitPosition, float handleThreshold) {
    if (rigidMode) {
        startGrab(hitPosition);
        currentGrabMode = GRAB_NORMAL;
        return;
    }

    if (tryStartGrabHandleGroup(hitPosition, handleThreshold)) {
        currentGrabMode = GRAB_HANDLE_GROUP;
    } else {
        startGrab(hitPosition);
        currentGrabMode = GRAB_NORMAL;
        std::cout << "Normal grab at position: " << hitPosition.x
                  << ", " << hitPosition.y << ", " << hitPosition.z << std::endl;
    }
}

void SoftBody::smartMove(const glm::vec3& newPosition, const glm::vec3& velocity) {
    if (rigidMode) {
        moveGrabbed(newPosition, velocity);
        return;
    }

    switch (currentGrabMode) {
    case GRAB_HANDLE_GROUP:
        moveGrabbedHandleGroup(newPosition, velocity);
        break;
    case GRAB_NORMAL:
        moveGrabbed(newPosition, velocity);
        break;
    default:
        break;
    }
}

void SoftBody::smartEndGrab(const glm::vec3& position, const glm::vec3& velocity) {
    if (rigidMode) {
        endGrab(position, velocity);
        currentGrabMode = GRAB_NORMAL;
        return;
    }

    switch (currentGrabMode) {
    case GRAB_HANDLE_GROUP:
        endGrabHandleGroup(position, velocity);
        break;
    case GRAB_NORMAL:
        endGrab(position, velocity);
        break;
    default:
        break;
    }
    currentGrabMode = GRAB_NONE;
}

void SoftBody::fullReset() {
    currentGrabMode = GRAB_NONE;
    activeHandleGroup = -1;
    grabId = -1;
    grabInvMass = 0.0f;

    handleGroups.clear();

    std::fill(velocities.begin(), velocities.end(), 0.0f);

    positions = meshData.verts;
    prevPositions = meshData.verts;

    for (size_t i = 0; i < vismeshDataArray.size(); i++) {
        vis_positions_array[i] = vismeshDataArray[i].verts;
    }

    std::fill(edgeLambdas.begin(), edgeLambdas.end(), 0.0f);
    std::fill(volLambdas.begin(), volLambdas.end(), 0.0f);

    initPhysics();

    updateTetMeshes();
    updateVisMeshes();

    std::cout << "Full reset completed - all data restored from original" << std::endl;
}
