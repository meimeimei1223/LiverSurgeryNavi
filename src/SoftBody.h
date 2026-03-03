#ifndef SOFT_BODY_H
#define SOFT_BODY_H

#include "VectorMath.h"
#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include "Hash.h"
#include <chrono>

class ShaderProgram;

class SoftBody {
public:
    struct MeshData {
        std::vector<float> verts;
        std::vector<int> tetIds;
        std::vector<int> tetEdgeIds;
        std::vector<int> tetSurfaceTriIds;
    };

    ~SoftBody();

    SoftBody(const MeshData& tetMesh,
             const std::vector<MeshData>& visMeshes,
             float edgeCompliance = 100.0f,
             float volCompliance = 0.0f);

    void preSolve(float dt, const glm::vec3& gravity);
    void solve(float dt);
    void postSolve(float dt);
    void initPhysics();

    void drawTetMesh(ShaderProgram& shader);
    void updateTetMeshes();

    float getTetVolume(int nr);
    void solveEdges(float compliance, float dt);
    void solveVolumes(float compliance, float dt);
    void setupTetMesh();
    void updateVisibility();
    void deleteBuffers();

    void startGrab(const glm::vec3& pos);
    void moveGrabbed(const glm::vec3& pos, const glm::vec3& vel);
    void endGrab(const glm::vec3& pos, const glm::vec3& vel);

    const std::vector<float>& getPositions() const { return positions; }
    const std::vector<int>& getTetIds() const { return tetIds; }
    const std::vector<int>& gettetSurfaceTriIds() const { return tetSurfaceTriIds; }

    const std::vector<float>& getVisPositions(int ids) const { return vis_positions_array[ids]; }
    const std::vector<int>& getVisSurfaceTriIds(int ids) const { return visSurfaceTriIds_array[ids]; }

    size_t getNumParticles() const { return numParticles; }
    size_t getNumVisParticles() const { return numVisParticles; }
    void setEdgeCompliance(float compliance) { edgeCompliance = compliance; }
    void setVolCompliance(float compliance) { volCompliance = compliance; }
    const MeshData& getMeshData() const { return meshData; }
    const MeshData& getVisMeshData(int ids) const { return vismeshDataArray[ids]; }
    void applyShapeRestoration(float strength);
    void reapplyHandleConstraints();

    std::vector<int> grabbedVertices;

    const glm::mat4& getModelMatrix() const { return modelMatrix; }
    void setModelMatrix(const glm::mat4& matrix) { modelMatrix = matrix; }

    static MeshData loadTetMesh(const std::string& filename);

    bool isTetMeshVisible() const {
        return showTetMesh;
    }

    const size_t& getNumVis() const { return numVisVerts; }

    void setupVisMeshes();
    void updateVisMeshes();
    void computeSkinningInfoForMesh(const std::vector<float>& visVerts,
                                    std::vector<float>& skinningInfoOut);
    void computeVisNormalsForMesh(size_t meshIdx);
    void draw_AllVisMeshes(ShaderProgram& shader,glm::vec3 cameraPos,std::vector<glm::vec4>& meshColors);

    std::vector<float> edgeLambdas;
    std::vector<float> volLambdas;

private:
    std::vector<int> activeParticles;
    const int NUM_ACTIVE_PARTICLES = 100;

    bool showTetMesh = true;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    const MeshData meshData;
    size_t numParticles;
    size_t numTets;

    size_t numVisVerts;

    std::vector<float> positions;
    std::vector<float> prevPositions;
    std::vector<float> velocities;
    std::vector<int> tetIds;
    std::vector<int> tetSurfaceTriIds;
    std::vector<int> edgeIds;
    std::vector<float> restVols;
    std::vector<float> edgeLengths;
    std::vector<float> invMasses;
    std::vector<float> oldInvMasses;

    size_t numVisParticles;
    GLuint tetVAO = 0;
    GLuint tetVBO = 0;
    GLuint tetEBO = 0;
    float edgeCompliance;
    float volCompliance;
    float damping;

    int grabId;
    float grabInvMass;

    std::vector<float> tempBuffer;
    std::vector<float> grads;

    const std::vector<std::vector<int>> volIdOrder = {
        {1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}
    };

public:
    std::vector<MeshData> vismeshDataArray;

    std::vector<std::vector<float>> vis_positions_array;
    std::vector<std::vector<int>> visSurfaceTriIds_array;
    std::vector<std::vector<float>> skinningInfo_array;

public:
    std::vector<GLuint> visVAOs;
    std::vector<GLuint> visVBOs;
    std::vector<GLuint> visEBOs;
    std::vector<GLuint> visNormalVBOs;

    int activeVisMeshIndex = 0;

    std::vector<std::vector<float>> original_vis_positions_array;
    std::vector<std::vector<float>> vis_normals_array;

    std::vector<float> originalPositions;
    std::vector<float> originalVisPositions;

public:
    struct HandleGroup {
        int centerVertex;
        std::vector<int> vertices;
        glm::vec3 centerPosition;
        std::vector<glm::vec3> relativePositions;
        float radius;

        void updateCenterPosition(const std::vector<float>& positions);
        void storeRelativePositions(const std::vector<float>& positions);
    };

    enum GrabMode {
        GRAB_NONE,
        GRAB_NORMAL,
        GRAB_HANDLE_GROUP
    };

    static const int MAX_HANDLE_GROUPS = 5;

    std::vector<HandleGroup> handleGroups;
    GrabMode currentGrabMode = GRAB_NONE;

    int findClosestVertex(const glm::vec3& position);
    glm::vec3 getVertexPosition(int index) const;

    void addHandleVertex(const glm::vec3& position);
    void addHandleVertexByIndex(int vertexId);
    void clearHandles();

    bool createHandleGroup(const glm::vec3& position, float radius = 1.0f);
    void clearHandleGroups();
    int findHandleGroupAtPosition(const glm::vec3& position, float threshold = 0.5f);
    std::vector<glm::vec3> getHandleGroupPositions(int groupIndex) const;

    void startGrabHandleGroup(const glm::vec3& hitPosition);
    bool tryStartGrabHandleGroup(const glm::vec3& hitPosition, float threshold = 0.5f);
    void moveGrabbedHandleGroup(const glm::vec3& newPosition, const glm::vec3& velocity);
    void endGrabHandleGroup(const glm::vec3& position, const glm::vec3& velocity);

    void smartGrab(const glm::vec3& hitPosition, float handleThreshold = 0.5f);
    void smartMove(const glm::vec3& newPosition, const glm::vec3& velocity);
    void smartEndGrab(const glm::vec3& position, const glm::vec3& velocity);

    void fullReset();

private:
    std::vector<int> handleVertices;
    int activeHandleGroup = -1;
    glm::vec3 grabOffset;

    bool rigidMode = false;
    bool isDragging = false;
    glm::vec3 rigidPosition;
    glm::quat rigidRotation;
    glm::vec3 rigidVelocity;
    glm::vec3 rigidAngularVelocity;
    std::vector<glm::vec3> restPositionsRelative;
public:
    void setRigidMode(bool enabled);
    bool isRigidMode() const { return rigidMode; }
    void updateRigidTransform();
    void rigidTranslate(const glm::vec3& delta);
    void rigidRotateAroundCenter(const glm::vec3& axis, float angle);
    glm::vec3 getRigidPosition() const { return rigidPosition; }

};
#endif
