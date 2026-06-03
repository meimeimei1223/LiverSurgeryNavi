#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Eigen/Dense>

#include "mCutMesh.h"

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
    // The Windows SDK #defines some of these names (notably REGISTERED) as
    // macros, which corrupts the enum on MSVC ("missing '}' before constant").
    // Undef them here so the enum parses regardless of include order. Undefining
    // a name that isn't a macro is a harmless no-op.
#ifdef _WIN32
#undef IDLE
#undef SELECTING_BOARD_POINTS
#undef SELECTING_OBJECT_POINTS
#undef READY_TO_REGISTER
#undef REGISTERED
#undef REFINING
#endif
    enum State {
        IDLE,
        SELECTING_BOARD_POINTS,
        SELECTING_OBJECT_POINTS,
        READY_TO_REGISTER,
        REGISTERED,
        REFINING
    };

    // ----------------------------------------------------------------------
    //  InitRotPreset (Phase 2: 幾何ベース; チャット 11 で 3×5 拡張)
    //  ---------------------------------------------------------------------
    //  15 プリセットは「足側から見た view」(caudal フル可視) をベースに、
    //  3×5 グリッドで上下左右にタイルト。
    //    - 内側 3×3 (id 0..8): ±20° 標準タイルト (チャット 10 までの 9 個)
    //    - 外側 3+3 (id 9..14, *_FAR): ±40° 拡張タイルト
    //      横方向 (dx) のみ ±2 倍 (= 40°)、縦方向 (dy) は ±20° のまま。
    //
    //  グリッドレイアウト (画面表示順、ボタンの位置 = 結果の見え方):
    //    +-------+------+------+------+-------+
    //    | 9     | 2    | 1    | 8    | 12    |     上段 (cranial 多め)
    //    | Up-R+ | Up-R | Up   | Up-L | Up-L+ |
    //    +-------+------+------+------+-------+
    //    | 10    | 3    | 0    | 7    | 13    |     中段
    //    | Right+| Right| Base | Left | Left+ |
    //    +-------+------+------+------+-------+
    //    | 11    | 4    | 5    | 6    | 14    |     下段 (caudal 多め)
    //    | Dn-R+ | Dn-R | Down | Dn-L | Dn-L+ |
    //    +-------+------+------+------+-------+
    //
    //  Radiology 慣例: 患者の右 (Right/Right+) は画面の左側に配置。
    //  各プリセットは肝臓自身の解剖軸 (d_LR, d_CC) を基準に計算されるので、
    //  患者が変わっても同じプリセットボタンが同じ解剖学的意味を持つ。
    //  CSV 互換: 旧エントリ 0..8 はそのまま、新規 9..14 が末尾に追加。
    // ----------------------------------------------------------------------
    enum InitRotPreset {
        PRESET_BASE          = 0,   // 中央: 足側から見上げる view (caudal フル可視)
        PRESET_BASE_UP       = 1,   // 上に ±20° 振る (cranial が見え始める)
        PRESET_BASE_UP_R     = 2,   // 上 + 右 (右葉が手前)
        PRESET_BASE_R        = 3,   // 右に振る
        PRESET_BASE_DN_R     = 4,   // 下 + 右 (caudal がより手前)
        PRESET_BASE_DN       = 5,   // 下に振る
        PRESET_BASE_DN_L     = 6,   // 下 + 左
        PRESET_BASE_L        = 7,   // 左に振る (左葉が手前)
        PRESET_BASE_UP_L     = 8,   // 上 + 左
        // --- チャット 11 拡張: 横方向 ±40° (dx=±2)、縦は ±20° のまま ---
        PRESET_BASE_UP_R_FAR = 9,   // 上 + 右(強) — 右葉がさらに手前
        PRESET_BASE_R_FAR    = 10,  // 右(強)に振る
        PRESET_BASE_DN_R_FAR = 11,  // 下 + 右(強)
        PRESET_BASE_UP_L_FAR = 12,  // 上 + 左(強)
        PRESET_BASE_L_FAR    = 13,  // 左(強)に振る — 左葉がさらに手前
        PRESET_BASE_DN_L_FAR = 14,  // 下 + 左(強)
        PRESET_COUNT
    };

    static const char* presetName(InitRotPreset p) {
        switch (p) {
        case PRESET_BASE:          return "Base";
        case PRESET_BASE_UP:       return "Up";
        case PRESET_BASE_UP_R:     return "Up-R";
        case PRESET_BASE_R:        return "Right";
        case PRESET_BASE_DN_R:     return "Dn-R";
        case PRESET_BASE_DN:       return "Down";
        case PRESET_BASE_DN_L:     return "Dn-L";
        case PRESET_BASE_L:        return "Left";
        case PRESET_BASE_UP_L:     return "Up-L";
        // --- *_FAR (チャット 11、横方向 ±40°) ---
        case PRESET_BASE_UP_R_FAR: return "Up-R+";
        case PRESET_BASE_R_FAR:    return "Right+";
        case PRESET_BASE_DN_R_FAR: return "Dn-R+";
        case PRESET_BASE_UP_L_FAR: return "Up-L+";
        case PRESET_BASE_L_FAR:    return "Left+";
        case PRESET_BASE_DN_L_FAR: return "Dn-L+";
        default:                   return "Unknown";
        }
    }

    InitRotPreset initRotPreset = PRESET_BASE;

    // ----------------------------------------------------------------------
    //  InitRotPosition (Phase 2 拡張: 重心の画面配置)
    //  ---------------------------------------------------------------------
    //  9 つの回転プリセットとは独立に、肝臓の重心の画面上の位置を 3 段階で
    //  選べる。Radiology 慣例:
    //    POS_CENTER: 画面中央 (デフォルト)
    //    POS_RIGHT : 患者の右 (= 画面の左) に重心を寄せる
    //    POS_LEFT  : 患者の左 (= 画面の右) に重心を寄せる
    //
    //  9 (回転) × 3 (位置) = 27 通りの初期姿勢が表現可能。
    //
    //  Phase 2 拡張: target cloud の AABB を 3 通り (全体/右半分/左半分) に
    //  分割し、Position 選択時に対応する subset AABB に合わせて liver mesh を
    //  正確にスケール + 配置する。getPresetRotation には scale_factor と
    //  position_translation (vec3) を呼び出し側 (main.cpp::applyInitRotation)
    //  で計算して渡す。
    // ----------------------------------------------------------------------
    enum InitRotPosition {
        POS_CENTER = 0,    // 中央 (デフォルト)
        POS_RIGHT  = 1,    // 患者の右 → 画面左に重心
        POS_LEFT   = 2,    // 患者の左 → 画面右に重心
        POS_COUNT
    };

    static const char* positionName(InitRotPosition p) {
        switch (p) {
        case POS_CENTER: return "Center";
        case POS_RIGHT:  return "Right";
        case POS_LEFT:   return "Left";
        default:         return "Unknown";
        }
    }

    InitRotPosition initRotPosition = POS_CENTER;

    State state = IDLE;
    std::vector<glm::vec3> boardPoints;
    std::vector<glm::vec3> objectPoints;

    glm::mat4 registrationMatrix = glm::mat4(1.0f);
    glm::mat4 appliedTransform    = glm::mat4(1.0f);
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
    float compRmse        = 0.0f;
    float compAvgError    = 0.0f;
    float compMaxError    = 0.0f;   // Target -> Source max (one-sided)
    float compMaxS2T      = 0.0f;   // Source -> Target max (other side)
    float compHausdorff   = 0.0f;   // max(compMaxError, compMaxS2T) = true Hausdorff (3D, scene units)
    float sil2DHausdorff  = 0.0f;   // 2D silhouette Hausdorff (pixels, two-sided)
    float compIoU2D       = 0.0f;   // 2D silhouette IoU (0..1, computed via fast rasterize step=8)
    int   compCount       = 0;
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
        compMaxS2T = 0.0f;
        compHausdorff = 0.0f;
        sil2DHausdorff = 0.0f;
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
        compMaxS2T = 0.0f;
        compHausdorff = 0.0f;
        sil2DHausdorff = 0.0f;
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

inline glm::vec3 computeMeshCentroidFromVertices(const std::vector<GLfloat>& verts) {
    glm::vec3 sum(0.0f);
    int count = (int)verts.size() / 3;
    if (count == 0) return sum;
    for (int i = 0; i < count; i++) {
        sum.x += verts[i * 3 + 0];
        sum.y += verts[i * 3 + 1];
        sum.z += verts[i * 3 + 2];
    }
    return sum / (float)count;
}

// =====================================================================
//  getPresetRotation (Phase 2: 幾何ベース)
//  ---------------------------------------------------------------------
//  9 プリセットの世界座標回転を「肝臓自身の解剖軸」基準で計算する。
//
//  引数:
//    preset   : InitRotPreset (PRESET_BASE..PRESET_BASE_UP_L)
//    centroid : 回転の pivot (通常は liverMesh の重心)
//    d_lr     : 肝臓 mesh の RIGHT 方向 (unit ベクトル、signed)
//               → LiverLeftRightLabel::Result::d_lr を渡す
//    d_cc     : 肝臓 mesh の CRANIAL 方向 (unit ベクトル、signed)
//               → LiverCranioCaudalLabel::Result::d_cc を渡す
//
//  処理:
//    1) R_base を構築: mesh の (d_LR, d_AP', d_CC) を world (+X, +Y, +Z)
//       に揃える正規直交回転 (d_AP' = cross(d_CC, d_LR) で右手系構成)。
//       これにより mesh の CRANIAL が world +Z (camera から遠方) に向き、
//       「足側から見上げる view」がベースになる。
//    2) Tilt offset: 3×3 grid の (dx, dy) ∈ {-1, 0, +1}² に応じて
//       world Y 軸まわり (左右) と world X 軸まわり (上下) に ±20° 回転。
//    3) Pivot 補正: centroid を中心に回転 (translate(c) * R * translate(-c))。
//
//  d_lr / d_cc が縮退している (zero / parallel) 場合は identity を返し、
//  registration を壊さないように安全側に倒す。呼び出し側で
//  g_liverLR / g_liverCC の valid() を確認してから呼ぶことを推奨。
//
//  使い方:
//    glm::vec3 c = computeMeshCentroidFromVertices(liverMesh3D->mVertices);
//    glm::mat4 R = getPresetRotation(preset, c,
//                                    g_liverLR.d_lr, g_liverCC.d_cc);
//    for (auto* m : organs) applyMatrixToMeshVerticesAndNormals(m, R);
// =====================================================================
inline glm::mat4 getPresetRotation(
    RegistrationData::InitRotPreset preset,
    const glm::vec3& centroid,
    const glm::vec3& d_lr,
    const glm::vec3& d_cc,
    float scale_factor = 1.0f,
    const glm::vec3& position_translation = glm::vec3(0.0f))
{
    // ---- 1) (dx, dy) を preset から決定 ----
    //   dx ∈ {-2, -1, 0, +1, +2}: 左右 (world Y 軸まわり)
    //      ±1 = 標準 ±20°、±2 = *_FAR の ±40° 拡張 (チャット 11)
    //   dy ∈ {-1, 0, +1}: 上下 (world X 軸まわり)、*_FAR でも値域は不変
    //   Base は (0, 0)。
    float dx = 0.0f, dy = 0.0f;
    switch (preset) {
    case RegistrationData::PRESET_BASE:          dx =  0.0f; dy =  0.0f; break;
    case RegistrationData::PRESET_BASE_UP:       dx =  0.0f; dy = +1.0f; break;
    case RegistrationData::PRESET_BASE_UP_R:     dx = +1.0f; dy = +1.0f; break;
    case RegistrationData::PRESET_BASE_R:        dx = +1.0f; dy =  0.0f; break;
    case RegistrationData::PRESET_BASE_DN_R:     dx = +1.0f; dy = -1.0f; break;
    case RegistrationData::PRESET_BASE_DN:       dx =  0.0f; dy = -1.0f; break;
    case RegistrationData::PRESET_BASE_DN_L:     dx = -1.0f; dy = -1.0f; break;
    case RegistrationData::PRESET_BASE_L:        dx = -1.0f; dy =  0.0f; break;
    case RegistrationData::PRESET_BASE_UP_L:     dx = -1.0f; dy = +1.0f; break;
    // --- *_FAR: 横 ±40° (dx=±2)、縦は ±20° のまま (dy=±1) ---
    case RegistrationData::PRESET_BASE_UP_R_FAR: dx = +2.0f; dy = +1.0f; break;
    case RegistrationData::PRESET_BASE_R_FAR:    dx = +2.0f; dy =  0.0f; break;
    case RegistrationData::PRESET_BASE_DN_R_FAR: dx = +2.0f; dy = -1.0f; break;
    case RegistrationData::PRESET_BASE_UP_L_FAR: dx = -2.0f; dy = +1.0f; break;
    case RegistrationData::PRESET_BASE_L_FAR:    dx = -2.0f; dy =  0.0f; break;
    case RegistrationData::PRESET_BASE_DN_L_FAR: dx = -2.0f; dy = -1.0f; break;
    default: break;
    }

    // ---- 2) d_lr / d_cc の sanity check ----
    const float lr_len = glm::length(d_lr);
    const float cc_len = glm::length(d_cc);
    if (lr_len < 1e-6f || cc_len < 1e-6f) {
        std::cerr << "[InitRot] getPresetRotation: d_lr or d_cc is zero. "
                  << "Returning identity. (Did you call recomputeLiverCC() / "
                     "recomputeLiverLR() first?)" << std::endl;
        return glm::mat4(1.0f);
    }
    glm::vec3 e1 = d_lr / lr_len;
    glm::vec3 e3 = d_cc / cc_len;
    glm::vec3 e2 = glm::cross(e3, e1);
    const float e2_len = glm::length(e2);
    if (e2_len < 1e-6f) {
        std::cerr << "[InitRot] getPresetRotation: d_lr and d_cc are parallel "
                  << "(cross product near zero). Returning identity." << std::endl;
        return glm::mat4(1.0f);
    }
    e2 = e2 / e2_len;
    // d_lr と d_cc は厳密には直交しない (PCA の異なる固有ベクトル
    // ペアで近似直交) ので、e1 を再構築して直交性を保証する。
    e1 = glm::cross(e2, e3);

    // ---- 3) R_base: mesh-frame の点を world-frame に揃える正規直交回転 ----
    //
    //   world_x = (mesh_point) · e1   (RIGHT 方向 → +X)
    //   world_y = (mesh_point) · e2   (構成された UP 方向 → +Y)
    //   world_z = (mesh_point) · e3   (CRANIAL 方向 → +Z, camera から遠方)
    //
    //   行列としては rows = (e1, e2, e3) の 3x3 を 4x4 に埋める。
    //   glm::mat4 は column-major (M[col][row]) なので、明示的に成分を置く。
    glm::mat4 R_base(1.0f);
    R_base[0] = glm::vec4(e1.x, e2.x, e3.x, 0.0f);   // col 0
    R_base[1] = glm::vec4(e1.y, e2.y, e3.y, 0.0f);   // col 1
    R_base[2] = glm::vec4(e1.z, e2.z, e3.z, 0.0f);   // col 2
    R_base[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    // ---- 4) Tilt offset (world 軸まわり) ----
    //   UI ボタンの直感に合わせるため、Up/Down と Right/Left で符号の扱いが異なる:
    //     dy > 0 ("Up"):    mesh を world X 軸まわりに -20° 回転 →
    //                       cranial 側 (world +Z) の頂点が画面上に来る
    //     dx > 0 ("Right"): mesh を world Y 軸まわりに +20° 回転 →
    //                       右葉 (world +X) が camera に近づき、画面右で大きく見える
    //   kTiltDeg = 20° 固定。dx の値域:
    //     ±1 = 標準プリセット (±20° 回転)
    //     ±2 = *_FAR プリセット (±40° 回転、チャット 11 で追加)
    //   dy は ±1 のまま (上下は標準角)。
    //
    //   実装注:
    //   - dx は元の符号 (+) のまま: camera が +Z 方向を向いている設定では
    //     Ry(+α) で mesh の右側が camera 近くに引き寄せられる。
    //   - dy は反転 (-) する: Rx(+α) では cranial 側が画面下に行ってしまうため、
    //     直感に合わせて符号を反転。
    const float kTiltDeg = 20.0f;
    const float kTiltRad = glm::radians(kTiltDeg);
    glm::mat4 Ry = glm::rotate(glm::mat4(1.0f),
                               dx * kTiltRad,        // Right/Left: 元の符号
                               glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 Rx = glm::rotate(glm::mat4(1.0f),
                               -dy * kTiltRad,       // Up/Down: 符号反転
                               glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 R_offset = Rx * Ry;   // Y 先、X 後 (慣例)

    // ---- 5) Pivot 補正 (centroid を中心に回転 + 重心スケール) ----
    glm::mat4 toOrigin   = glm::translate(glm::mat4(1.0f), -centroid);
    glm::mat4 fromOrigin = glm::translate(glm::mat4(1.0f),  centroid);

    // ---- 6) Scale factor (Position 用に肝臓全体を縮小・拡大) ----
    //   target cloud の subset AABB (Position に応じた半分 or 全体) と
    //   target cloud 全体の AABB の比率 (subset_diag / full_diag) を呼び出し
    //   側で渡す。POS_CENTER の場合は 1.0 で素通し。
    //   Scale も centroid 中心で適用する (T(c) * S * T(-c))。
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(scale_factor));

    // ---- 7) Position translation (重心を target subset の center に移動) ----
    //   呼び出し側で「target_subset.center - target_full.center」を渡す。
    //   POS_CENTER の場合は (0,0,0) で素通し。
    //   AABB ベースの正確な平行移動なので、前回の bbox_diag*0.15 ヒューリスティック
    //   は廃止された。
    glm::mat4 T_pos = glm::translate(glm::mat4(1.0f), position_translation);

    // 最終 transform:
    //   1. T(-centroid): 重心を原点に移動
    //   2. S:            scale (Position 用)
    //   3. R_base * R_offset: 幾何ベース回転 + tilt
    //   4. T(+centroid): 重心位置に戻す
    //   5. T_pos:        重心を target subset center にシフト
    return T_pos * fromOrigin * R_offset * R_base * S * toOrigin;
}

inline void applyMatrixToMeshVerticesAndNormals(
    mCutMesh* mesh,
    const glm::mat4& transform)
{
    if (!mesh || mesh->mVertices.empty()) return;

    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));

    for (size_t i = 0; i + 2 < mesh->mVertices.size(); i += 3) {
        glm::vec4 v(mesh->mVertices[i],
                    mesh->mVertices[i+1],
                    mesh->mVertices[i+2], 1.0f);
        v = transform * v;
        mesh->mVertices[i]   = v.x;
        mesh->mVertices[i+1] = v.y;
        mesh->mVertices[i+2] = v.z;
    }

    if (!mesh->mNormals.empty()) {
        for (size_t i = 0; i + 2 < mesh->mNormals.size(); i += 3) {
            glm::vec3 n(mesh->mNormals[i],
                        mesh->mNormals[i+1],
                        mesh->mNormals[i+2]);
            n = glm::normalize(normalMatrix * n);
            mesh->mNormals[i]   = n.x;
            mesh->mNormals[i+1] = n.y;
            mesh->mNormals[i+2] = n.z;
        }
    }
}
