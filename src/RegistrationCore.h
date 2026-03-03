#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <Eigen/Dense>

namespace Reg3D {

struct UmeyamaResult {
    glm::mat4 transformation;
    float scale;
    int correspondenceCount;
    float averageError;
    bool success;

    UmeyamaResult()
        : transformation(1.0f)
        , scale(1.0f)
        , correspondenceCount(0)
        , averageError(0.0f)
        , success(false) {}
};

inline int g_maxCorrespondenceMarkers = 20;

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() : min(FLT_MAX), max(-FLT_MAX) {}

    void expand(const glm::vec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    void expand(const AABB& box) {
        min = glm::min(min, box.min);
        max = glm::max(max, box.max);
    }

    glm::vec3 center() const {
        return (min + max) * 0.5f;
    }

    bool intersectRay(const glm::vec3& origin, const glm::vec3& direction) const {
        glm::vec3 invDir = 1.0f / direction;
        glm::vec3 t0 = (min - origin) * invDir;
        glm::vec3 t1 = (max - origin) * invDir;

        glm::vec3 tmin = glm::min(t0, t1);
        glm::vec3 tmax = glm::max(t0, t1);

        float tNear = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
        float tFar = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

        return tNear <= tFar && tFar >= 0.0f;
    }
};

struct BVHNode {
    AABB bbox;
    int leftChild;
    int rightChild;
    int triangleStart;
    int triangleCount;

    BVHNode() : leftChild(-1), rightChild(-1), triangleStart(-1), triangleCount(0) {}

    bool isLeaf() const {
        return leftChild == -1 && rightChild == -1;
    }
};

struct Triangle {
    glm::vec3 v0, v1, v2;
    int originalIndex;
};

class BVHTree {
public:
    std::vector<BVHNode> nodes;
    std::vector<Triangle> triangles;

    void build(const std::vector<GLfloat>& vertices, const std::vector<GLuint>& indices) {
        triangles.clear();
        nodes.clear();

        for (size_t i = 0; i < indices.size(); i += 3) {
            Triangle tri;
            tri.v0 = glm::vec3(vertices[indices[i] * 3],
                               vertices[indices[i] * 3 + 1],
                               vertices[indices[i] * 3 + 2]);
            tri.v1 = glm::vec3(vertices[indices[i + 1] * 3],
                               vertices[indices[i + 1] * 3 + 1],
                               vertices[indices[i + 1] * 3 + 2]);
            tri.v2 = glm::vec3(vertices[indices[i + 2] * 3],
                               vertices[indices[i + 2] * 3 + 1],
                               vertices[indices[i + 2] * 3 + 2]);
            tri.originalIndex = i / 3;
            triangles.push_back(tri);
        }

        if (triangles.empty()) return;

        nodes.reserve(triangles.size() * 2);
        buildRecursive(0, triangles.size(), 0);
    }

    int countIntersections(const glm::vec3& origin, const glm::vec3& direction) const {
        if (nodes.empty()) return 0;
        return countIntersectionsRecursive(0, origin, direction);
    }

private:
    int buildRecursive(int start, int end, int depth) {
        int nodeIdx = nodes.size();
        nodes.emplace_back();
        BVHNode& node = nodes[nodeIdx];

        for (int i = start; i < end; i++) {
            node.bbox.expand(triangles[i].v0);
            node.bbox.expand(triangles[i].v1);
            node.bbox.expand(triangles[i].v2);
        }

        int count = end - start;

        if (count <= 4 || depth > 20) {
            node.triangleStart = start;
            node.triangleCount = count;
            return nodeIdx;
        }

        glm::vec3 extent = node.bbox.max - node.bbox.min;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        int mid = (start + end) / 2;
        std::nth_element(triangles.begin() + start,
                         triangles.begin() + mid,
                         triangles.begin() + end,
                         [axis](const Triangle& a, const Triangle& b) {
                             glm::vec3 ca = (a.v0 + a.v1 + a.v2) / 3.0f;
                             glm::vec3 cb = (b.v0 + b.v1 + b.v2) / 3.0f;
                             return ca[axis] < cb[axis];
                         });

        node.leftChild = buildRecursive(start, mid, depth + 1);
        node.rightChild = buildRecursive(mid, end, depth + 1);

        return nodeIdx;
    }

    int countIntersectionsRecursive(int nodeIdx, const glm::vec3& origin, const glm::vec3& direction) const {
        const BVHNode& node = nodes[nodeIdx];

        if (!node.bbox.intersectRay(origin, direction)) {
            return 0;
        }

        if (node.isLeaf()) {
            int count = 0;
            const float EPSILON = 0.0000001f;

            for (int i = 0; i < node.triangleCount; i++) {
                const Triangle& tri = triangles[node.triangleStart + i];

                glm::vec3 edge1 = tri.v1 - tri.v0;
                glm::vec3 edge2 = tri.v2 - tri.v0;
                glm::vec3 h = glm::cross(direction, edge2);
                float a = glm::dot(edge1, h);

                if (a > -EPSILON && a < EPSILON) continue;

                float f = 1.0f / a;
                glm::vec3 s = origin - tri.v0;
                float u = f * glm::dot(s, h);

                if (u < 0.0f || u > 1.0f) continue;

                glm::vec3 q = glm::cross(s, edge1);
                float v = f * glm::dot(direction, q);

                if (v < 0.0f || u + v > 1.0f) continue;

                float t = f * glm::dot(edge2, q);

                if (t > EPSILON) {
                    count++;
                }
            }
            return count;
        }

        int leftCount = countIntersectionsRecursive(node.leftChild, origin, direction);
        int rightCount = countIntersectionsRecursive(node.rightChild, origin, direction);

        return leftCount + rightCount;
    }
};

struct SurfaceVertexCache {
    std::vector<std::vector<size_t>> surfaceVertexIndices;
    std::vector<std::string> cachedMeshNames;
    std::vector<BVHTree> bvhTrees;
    bool isValid;
    size_t meshCount;

    SurfaceVertexCache() : isValid(false), meshCount(0) {}
};

static SurfaceVertexCache g_surfaceCache;

inline void clearSurfaceVertexCache() {
    g_surfaceCache.surfaceVertexIndices.clear();
    g_surfaceCache.cachedMeshNames.clear();
    g_surfaceCache.bvhTrees.clear();
    g_surfaceCache.isValid = false;
    g_surfaceCache.meshCount = 0;
    std::cout << "Surface vertex cache cleared" << std::endl;
}

inline bool isCacheValid(const std::vector<std::string>& meshNames) {
    if (!g_surfaceCache.isValid) return false;
    if (g_surfaceCache.meshCount != meshNames.size()) return false;
    if (g_surfaceCache.cachedMeshNames.size() != meshNames.size()) return false;

    for (size_t i = 0; i < meshNames.size(); i++) {
        if (g_surfaceCache.cachedMeshNames[i] != meshNames[i]) {
            return false;
        }
    }

    return true;
}

inline bool isPointInsideMeshUsingBVH(
    const glm::vec3& point,
    const BVHTree& bvh) {

    std::vector<glm::vec3> test_directions = {
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f))
    };

    int internal_votes = 0;

    for (const auto& direction : test_directions) {
        int intersection_count = bvh.countIntersections(point, direction);

        if ((intersection_count % 2) == 1) {
            internal_votes++;
        }
    }

    return internal_votes > (int)(test_directions.size() / 2);
}

inline glm::mat4 UmeyamaRegistration(
    const std::vector<glm::vec3>& source,
    const std::vector<glm::vec3>& target
    ) {
    int n = source.size();

    glm::vec3 sourceCentroid(0), targetCentroid(0);
    for (int i = 0; i < n; i++) {
        sourceCentroid += source[i];
        targetCentroid += target[i];
    }
    sourceCentroid /= n;
    targetCentroid /= n;

    Eigen::MatrixXf X(3, n);
    Eigen::MatrixXf Y(3, n);

    for (int i = 0; i < n; i++) {
        glm::vec3 xs = source[i] - sourceCentroid;
        glm::vec3 yt = target[i] - targetCentroid;

        X(0, i) = xs.x;
        X(1, i) = xs.y;
        X(2, i) = xs.z;

        Y(0, i) = yt.x;
        Y(1, i) = yt.y;
        Y(2, i) = yt.z;
    }

    float varX = 0;
    for (int i = 0; i < n; i++) {
        varX += X.col(i).squaredNorm();
    }
    varX /= n;

    Eigen::Matrix3f C = (1.0f / n) * Y * X.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Vector3f S = svd.singularValues();
    Eigen::Matrix3f V = svd.matrixV();

    Eigen::Matrix3f D = Eigen::Matrix3f::Identity();
    float det = U.determinant() * V.determinant();
    if (det < 0) {
        D(2, 2) = -1;
    }

    Eigen::Matrix3f R = U * D * V.transpose();

    float scale = 0;
    for (int i = 0; i < 3; i++) {
        scale += S(i) * D(i, i);
    }
    scale /= varX;

    Eigen::Vector3f srcC(sourceCentroid.x, sourceCentroid.y, sourceCentroid.z);
    Eigen::Vector3f tgtC(targetCentroid.x, targetCentroid.y, targetCentroid.z);
    Eigen::Vector3f t = tgtC - scale * R * srcC;

    glm::mat4 transform(1.0f);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            transform[j][i] = scale * R(i, j);
        }
    }

    transform[3][0] = t(0);
    transform[3][1] = t(1);
    transform[3][2] = t(2);

    return transform;
}

inline float estimateScaleOnly(
    const std::vector<glm::vec3>& source,
    const std::vector<glm::vec3>& target) {

    if (source.size() < 3 || source.size() != target.size()) {
        return 1.0f;
    }

    glm::vec3 src_centroid(0.0f);
    glm::vec3 tgt_centroid(0.0f);

    for (size_t i = 0; i < source.size(); i++) {
        src_centroid += source[i];
        tgt_centroid += target[i];
    }
    src_centroid /= source.size();
    tgt_centroid /= target.size();

    std::vector<float> ratios;

    for (size_t i = 0; i < source.size(); i++) {
        float src_dist = glm::length(source[i] - src_centroid);
        float tgt_dist = glm::length(target[i] - tgt_centroid);

        if (src_dist > 0.01f) {
            ratios.push_back(tgt_dist / src_dist);
        }
    }

    if (ratios.empty()) {
        return 1.0f;
    }

    std::sort(ratios.begin(), ratios.end());
    return ratios[ratios.size() / 2];
}

static const std::vector<glm::vec3> ICOSAHEDRON_DIRECTIONS = {
    glm::normalize(glm::vec3( 0.000f,  1.000f,  1.618f)),
    glm::normalize(glm::vec3( 0.000f,  1.000f, -1.618f)),
    glm::normalize(glm::vec3( 0.000f, -1.000f,  1.618f)),
    glm::normalize(glm::vec3( 0.000f, -1.000f, -1.618f)),
    glm::normalize(glm::vec3( 1.618f,  0.000f,  1.000f)),
    glm::normalize(glm::vec3( 1.618f,  0.000f, -1.000f)),
    glm::normalize(glm::vec3(-1.618f,  0.000f,  1.000f)),
    glm::normalize(glm::vec3(-1.618f,  0.000f, -1.000f)),
    glm::normalize(glm::vec3( 1.000f,  1.618f,  0.000f)),
    glm::normalize(glm::vec3( 1.000f, -1.618f,  0.000f)),
    glm::normalize(glm::vec3(-1.000f,  1.618f,  0.000f)),
    glm::normalize(glm::vec3(-1.000f, -1.618f,  0.000f))
};

inline glm::vec3 getClusterColor(int id) {
    const std::vector<glm::vec3> colors = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.5f, 0.0f},
        {0.5f, 0.0f, 1.0f},
        {0.5f, 1.0f, 0.0f},
        {1.0f, 0.0f, 0.5f},
        {0.0f, 0.5f, 1.0f},
        {0.5f, 0.5f, 0.5f}
    };
    return colors[id % colors.size()];
}

struct VertexCluster {
    int clusterId;
    glm::vec3 viewDirection;
    std::vector<int> visibleVertexIndices;
    std::vector<glm::vec3> visibleVertices;
    glm::vec3 clusterCentroid;

    VertexCluster() : clusterId(-1), clusterCentroid(0.0f) {}

    void computeCentroid() {
        if (visibleVertices.empty()) {
            clusterCentroid = glm::vec3(0.0f);
            return;
        }

        glm::vec3 sum(0.0f);
        for (const auto& v : visibleVertices) {
            sum += v;
        }
        clusterCentroid = sum / static_cast<float>(visibleVertices.size());
    }

};

struct RaycastClusteringResult {
    std::vector<VertexCluster> clusters;
    glm::vec3 meshCentroid;
    float meshScale;

    RaycastClusteringResult() : meshCentroid(0.0f), meshScale(0.0f) {}

    VertexCluster* getCluster(int id) {
        for (auto& cluster : clusters) {
            if (cluster.clusterId == id) {
                return &cluster;
            }
        }
        return nullptr;
    }

    void printStatistics() const {
        std::cout << "\n========== Raycast Clustering Statistics ==========\n";
        std::cout << "Mesh Centroid: " << meshCentroid.x << ", "
                  << meshCentroid.y << ", " << meshCentroid.z << "\n";
        std::cout << "Mesh Scale: " << meshScale << "\n";
        std::cout << "Number of Clusters: " << clusters.size() << "\n\n";

        for (const auto& cluster : clusters) {
            std::cout << "Cluster " << cluster.clusterId << ":\n";
            std::cout << "  Visible Vertices: " << cluster.visibleVertices.size() << "\n";
            std::cout << "  View Direction: (" << cluster.viewDirection.x << ", "
                      << cluster.viewDirection.y << ", " << cluster.viewDirection.z << ")\n";
            std::cout << "  Centroid: (" << cluster.clusterCentroid.x << ", "
                      << cluster.clusterCentroid.y << ", " << cluster.clusterCentroid.z << ")\n\n";
        }
        std::cout << "==================================================\n";
    }
};

class RaycastClusterer {
private:
    const BVHTree& bvh;

    bool rayTriangleIntersect(
        const glm::vec3& rayOrigin,
        const glm::vec3& rayDirection,
        const Triangle& tri,
        float& t) const {

        const float EPSILON = 1e-8f;

        glm::vec3 edge1 = tri.v1 - tri.v0;
        glm::vec3 edge2 = tri.v2 - tri.v0;
        glm::vec3 h = glm::cross(rayDirection, edge2);
        float a = glm::dot(edge1, h);

        if (std::abs(a) < EPSILON) {
            return false;
        }

        float f = 1.0f / a;
        glm::vec3 s = rayOrigin - tri.v0;
        float u = f * glm::dot(s, h);

        if (u < 0.0f || u > 1.0f) {
            return false;
        }

        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(rayDirection, q);

        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }

        t = f * glm::dot(edge2, q);

        return t > EPSILON;
    }

    bool raycast(
        const glm::vec3& rayOrigin,
        const glm::vec3& rayDirection,
        int& hitTriangleIndex,
        float& hitDistance) const {

        hitTriangleIndex = -1;
        hitDistance = FLT_MAX;

        std::vector<int> stack;
        stack.reserve(64);
        stack.push_back(0);

        while (!stack.empty()) {
            int nodeIndex = stack.back();
            stack.pop_back();

            const BVHNode& node = bvh.nodes[nodeIndex];

            if (!node.bbox.intersectRay(rayOrigin, rayDirection)) {
                continue;
            }

            if (node.isLeaf()) {

                for (int i = 0; i < node.triangleCount; i++) {
                    int triIndex = node.triangleStart + i;
                    const Triangle& tri = bvh.triangles[triIndex];

                    float t;
                    if (rayTriangleIntersect(rayOrigin, rayDirection, tri, t)) {
                        if (t < hitDistance) {
                            hitDistance = t;
                            hitTriangleIndex = triIndex;
                        }
                    }
                }
            } else {

                if (node.rightChild >= 0) {
                    stack.push_back(node.rightChild);
                }
                if (node.leftChild >= 0) {
                    stack.push_back(node.leftChild);
                }
            }
        }

        return hitTriangleIndex >= 0;
    }

public:
    RaycastClusterer(const BVHTree& bvhTree) : bvh(bvhTree) {}

    VertexCluster extractVisibleVertices(
        const glm::vec3& viewDirection,
        const glm::vec3& meshCentroid,
        float meshScale,
        const std::vector<GLfloat>& vertices,
        int clusterId) const {

        VertexCluster cluster;
        cluster.clusterId = clusterId;
        cluster.viewDirection = viewDirection;

        glm::vec3 cameraPos = meshCentroid - viewDirection * (meshScale * 3.0f);

        int vertexCount = vertices.size() / 3;

        for (int i = 0; i < vertexCount; i++) {
            glm::vec3 vertex(
                vertices[i * 3 + 0],
                vertices[i * 3 + 1],
                vertices[i * 3 + 2]
                );

            glm::vec3 rayDir = glm::normalize(vertex - cameraPos);

            int hitTriIndex;
            float hitDist;

            if (raycast(cameraPos, rayDir, hitTriIndex, hitDist)) {

                float distToVertex = glm::length(vertex - cameraPos);
                float distToHit = hitDist;

                const float EPSILON = 0.01f * meshScale;
                if (std::abs(distToHit - distToVertex) < EPSILON) {
                    cluster.visibleVertexIndices.push_back(i);
                    cluster.visibleVertices.push_back(vertex);
                }
            }
        }

        cluster.computeCentroid();

        return cluster;
    }

    RaycastClusteringResult performClustering(
        const std::vector<GLfloat>& vertices,
        const std::vector<GLuint>& indices) {

        RaycastClusteringResult result;

        result.meshCentroid = computeMeshCentroid(vertices);
        result.meshScale = computeMeshScale(vertices);

        std::cout << "Starting Raycast Clustering...\n";
        std::cout << "Mesh Centroid: " << result.meshCentroid.x << ", "
                  << result.meshCentroid.y << ", " << result.meshCentroid.z << "\n";
        std::cout << "Mesh Scale: " << result.meshScale << "\n";

        for (int i = 0; i < (int)ICOSAHEDRON_DIRECTIONS.size(); i++) {
            std::cout << "\nProcessing Cluster " << i << "...\n";

            VertexCluster cluster = extractVisibleVertices(
                ICOSAHEDRON_DIRECTIONS[i],
                result.meshCentroid,
                result.meshScale,
                vertices,
                i
                );

            std::cout << "  Visible Vertices: " << cluster.visibleVertices.size() << "\n";

            result.clusters.push_back(cluster);
        }

        std::cout << "\nRaycast Clustering Completed!\n";
        result.printStatistics();

        return result;
    }

private:

    glm::vec3 computeMeshCentroid(const std::vector<GLfloat>& vertices) const {
        glm::vec3 sum(0.0f);
        int count = vertices.size() / 3;

        for (int i = 0; i < count; i++) {
            sum.x += vertices[i * 3 + 0];
            sum.y += vertices[i * 3 + 1];
            sum.z += vertices[i * 3 + 2];
        }

        return sum / static_cast<float>(count);
    }

    float computeMeshScale(const std::vector<GLfloat>& vertices) const {
        glm::vec3 minBound(FLT_MAX);
        glm::vec3 maxBound(-FLT_MAX);

        int count = vertices.size() / 3;
        for (int i = 0; i < count; i++) {
            glm::vec3 v(
                vertices[i * 3 + 0],
                vertices[i * 3 + 1],
                vertices[i * 3 + 2]
                );
            minBound = glm::min(minBound, v);
            maxBound = glm::max(maxBound, v);
        }

        return glm::length(maxBound - minBound);
    }
};

}

struct RegistrationData {
    enum State {
        IDLE,
        SELECTING_BOARD_POINTS,
        SELECTING_OBJECT_POINTS,
        READY_TO_REGISTER,
        REGISTERED,
        REFINING
    };

    State state = IDLE;
    std::vector<glm::vec3> boardPoints;
    std::vector<glm::vec3> objectPoints;

    glm::mat4 registrationMatrix = glm::mat4(1.0f);
    bool useRegistration = false;
    float averageError = 0.0f;
    float rmse = 0.0f;
    float maxError = 0.0f;
    float scaleFactor = 1.0f;

    // ICP/FGR metrics (saved from registration pipeline)
    float fitness       = 0.0f;
    float icpRmse       = 0.0f;
    int   bestIteration = 0;

    // Refine tracking
    int   refineCount        = 0;
    float refineInitialRMSE  = 0.0f;
    float refineBestRMSE     = 0.0f;
    int   refineBestIteration = 0;

    // Unified comparison metrics (Liver visible vs depth, all correspondences, no sampling)
    float compRmse     = 0.0f;
    float compAvgError = 0.0f;
    float compMaxError = 0.0f;
    int   compCount    = 0;
    std::vector<glm::vec3> compSource;
    std::vector<glm::vec3> compTarget;



    int targetPointCount = 3;

    void reset() {
        boardPoints.clear();
        objectPoints.clear();
        state = IDLE;
        useRegistration = false;
        registrationMatrix = glm::mat4(1.0f);
        averageError = 0.0f;
        rmse = 0.0f;
        maxError = 0.0f;
        scaleFactor = 1.0f;
        fitness = 0.0f;
        icpRmse = 0.0f;
        bestIteration = 0;
        refineCount = 0;
        refineInitialRMSE = 0.0f;
        refineBestRMSE = 0.0f;
        refineBestIteration = 0;
        compRmse = 0.0f;
        compAvgError = 0.0f;
        compMaxError = 0.0f;
        compCount = 0;
        compSource.clear();
        compTarget.clear();
    }

    void resetTransformOnly() {
        state = IDLE;
        useRegistration = false;
        registrationMatrix = glm::mat4(1.0f);
        averageError = 0.0f;
        rmse = 0.0f;
        maxError = 0.0f;
        scaleFactor = 1.0f;
        fitness = 0.0f;
        icpRmse = 0.0f;
        bestIteration = 0;
        refineCount = 0;
        refineInitialRMSE = 0.0f;
        refineBestRMSE = 0.0f;
        refineBestIteration = 0;
        compRmse = 0.0f;
        compAvgError = 0.0f;
        compMaxError = 0.0f;
        compCount = 0;
        compSource.clear();
        compTarget.clear();
    }

    void clearPoints() {
        boardPoints.clear();
        objectPoints.clear();
    }

    bool canRegister() const {
        return boardPoints.size() >= 3 &&
               boardPoints.size() == objectPoints.size() &&
               boardPoints.size() == (size_t)targetPointCount;
    }
};
