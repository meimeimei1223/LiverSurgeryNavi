#pragma once

#include "RegistrationCore.h"
#include "nanoflann.hpp"
#include "mCutMesh.h"
#include "DepthUtils.h"

#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <numeric>
#include <functional>
#include <queue>
#include <random>
#include <map>
#include <set>

extern std::vector<glm::vec3> g_cluster1Points;
extern std::vector<glm::vec3> g_cluster2Points;
extern std::vector<glm::vec3> g_targetPoints;
extern bool g_showClusterVisualization;
extern RegistrationData registrationHandle;
extern void setUp(mCutMesh& srcMesh);
extern std::function<void(float, const char*)> g_progressCallback;

namespace Reg3DCustom {

struct PointCloud {
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> colors;
    std::vector<float>     boundaryDist;

    size_t size()  const { return points.size(); }
    bool   empty() const { return points.empty(); }

    void clear() {
        points.clear();
        normals.clear();
        colors.clear();
        boundaryDist.clear();
    }

    void reserve(size_t n) {
        points.reserve(n);
        normals.reserve(n);
        colors.reserve(n);
    }

    void addPoint(const glm::vec3& p) {
        points.push_back(p);
    }

    void addPointWithNormal(const glm::vec3& p, const glm::vec3& n) {
        points.push_back(p);
        normals.push_back(n);
    }

    void addPointFull(const glm::vec3& p, const glm::vec3& n, const glm::vec3& c) {
        points.push_back(p);
        normals.push_back(n);
        colors.push_back(c);
    }

    bool hasNormals()      const { return !normals.empty()      && normals.size()      == points.size(); }
    bool hasColors()       const { return !colors.empty()       && colors.size()        == points.size(); }
    bool hasBoundaryDist() const { return !boundaryDist.empty() && boundaryDist.size() == points.size(); }
};

struct RegistrationResult {
    glm::mat4 transformation = glm::mat4(1.0f);
    float fitness     = 0.0f;
    float inlier_rmse = 0.0f;
    std::vector<std::pair<int, int>> correspondences;
};

struct FeatureSet {
    std::vector<std::vector<float>> data;
    int dimension = 33;

    size_t numPoints() const { return data.size(); }

    void resize(int dim, int n) {
        dimension = dim;
        data.assign(n, std::vector<float>(dim, 0.0f));
    }
};

struct NanoflannAdaptor {
    const std::vector<glm::vec3>& pts;

    NanoflannAdaptor(const std::vector<glm::vec3>& points) : pts(points) {}

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx][static_cast<int>(dim)];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree3D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, NanoflannAdaptor>,
    NanoflannAdaptor,
    3
    >;

inline std::unique_ptr<KDTree3D> buildKDTree(
    const NanoflannAdaptor& adaptor,
    int maxLeafSize = 15)
{
    auto tree = std::make_unique<KDTree3D>(
        3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeafSize));
    tree->buildIndex();
    return tree;
}

inline bool searchKNN1(
    const KDTree3D& tree,
    const glm::vec3& query,
    size_t& out_index,
    float& out_dist_sq)
{
    const float queryPt[3] = { query.x, query.y, query.z };
    nanoflann::KNNResultSet<float> resultSet(1);
    resultSet.init(&out_index, &out_dist_sq);
    return tree.findNeighbors(resultSet, queryPt);
}

inline size_t searchKNN(
    const KDTree3D& tree,
    const glm::vec3& query,
    int k,
    std::vector<size_t>& out_indices,
    std::vector<float>& out_dists_sq)
{
    out_indices.resize(k);
    out_dists_sq.resize(k);

    const float queryPt[3] = { query.x, query.y, query.z };
    nanoflann::KNNResultSet<float> resultSet(k);
    resultSet.init(out_indices.data(), out_dists_sq.data());
    tree.findNeighbors(resultSet, queryPt);

    return resultSet.size();
}

inline size_t searchRadius(
    const KDTree3D& tree,
    const glm::vec3& query,
    float radius,
    std::vector<nanoflann::ResultItem<unsigned int, float>>& results)
{
    const float queryPt[3] = { query.x, query.y, query.z };
    float radius_sq = radius * radius;

    nanoflann::SearchParameters params;
    params.sorted = false;

    return tree.radiusSearch(queryPt, radius_sq, results, params);
}

struct NanoflannAdaptorD {
    const std::vector<glm::dvec3>& pts;

    NanoflannAdaptorD(const std::vector<glm::dvec3>& points) : pts(points) {}

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx][static_cast<int>(dim)];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree3DD = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, NanoflannAdaptorD>,
    NanoflannAdaptorD,
    3
    >;

inline std::unique_ptr<KDTree3DD> buildKDTreeD(
    const NanoflannAdaptorD& adaptor,
    int maxLeafSize = 15)
{
    auto tree = std::make_unique<KDTree3DD>(
        3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeafSize));
    tree->buildIndex();
    return tree;
}

inline bool searchKNN1D(
    const KDTree3DD& tree,
    const glm::dvec3& query,
    size_t& out_index,
    double& out_dist_sq)
{
    const double queryPt[3] = { query.x, query.y, query.z };
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&out_index, &out_dist_sq);
    return tree.findNeighbors(resultSet, queryPt);
}

inline void searchKNND(
    const KDTree3DD& tree,
    const glm::dvec3& query,
    int k,
    std::vector<size_t>& out_indices,
    std::vector<double>& out_dists_sq)
{
    out_indices.resize(k);
    out_dists_sq.resize(k);
    const double queryPt[3] = { query.x, query.y, query.z };
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(out_indices.data(), out_dists_sq.data());
    tree.findNeighbors(resultSet, queryPt);
    size_t found = resultSet.size();
    out_indices.resize(found);
    out_dists_sq.resize(found);
}

struct NanoflannFeatureAdaptor {
    const std::vector<std::vector<float>>& feats;
    int dim;

    NanoflannFeatureAdaptor(const std::vector<std::vector<float>>& features, int dimension)
        : feats(features), dim(dimension) {}

    inline size_t kdtree_get_point_count() const { return feats.size(); }

    inline float kdtree_get_pt(const size_t idx, const size_t d) const {
        return feats[idx][static_cast<int>(d)];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTreeFeature = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, NanoflannFeatureAdaptor>,
    NanoflannFeatureAdaptor,
    33
    >;

inline std::unique_ptr<KDTreeFeature> buildFeatureKDTree(
    const NanoflannFeatureAdaptor& adaptor,
    int maxLeafSize = 10)
{
    auto tree = std::make_unique<KDTreeFeature>(
        33, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeafSize));
    tree->buildIndex();
    return tree;
}

class NoOpen3DRegistration {
public:

    std::shared_ptr<PointCloud> extractFrontFacePoints(
        const mCutMesh& mesh,
        int gridWidth,
        int gridHeight,
        float z_threshold = 0.1f)
    {
        auto cloud = std::make_shared<PointCloud>();

        int frontVertexCount = (gridWidth + 1) * (gridHeight + 1);

        bool hasBdMap = ensureBoundaryMap();

        int maskFiltered = 0;
        int zFiltered = 0;

        for (int i = 0; i < frontVertexCount; i++) {
            float x = mesh.mVertices[i * 3];
            float y = mesh.mVertices[i * 3 + 1];
            float z = mesh.mVertices[i * 3 + 2];

            float bd = hasBdMap ? getBoundaryDistForGridVertex(i, gridWidth, gridHeight) : 9999.0f;
            bool insideMask = hasBdMap ? (bd < 9000.0f) : (z > z_threshold);

            if (insideMask) {
                glm::vec3 point(x, y, z);
                cloud->points.push_back(point);

                if (!mesh.mNormals.empty()) {
                    cloud->normals.push_back(glm::vec3(
                        mesh.mNormals[i * 3],
                        mesh.mNormals[i * 3 + 1],
                        mesh.mNormals[i * 3 + 2]
                        ));
                }

                cloud->boundaryDist.push_back(bd);
                if (hasBdMap) maskFiltered++;
            } else {
                if (!hasBdMap) zFiltered++;
            }
        }

        cloud->colors.resize(cloud->points.size(), glm::vec3(0.0f, 1.0f, 0.0f));

        std::cout << "  [Custom] Extracted " << cloud->points.size()
                  << " front face points";
        if (hasBdMap)
            std::cout << " (mask-based, boundaryDist: YES)";
        else
            std::cout << " (z-threshold=" << z_threshold << " fallback)";
        std::cout << std::endl;

        return cloud;
    }

    std::shared_ptr<PointCloud> voxelDownSample(
        std::shared_ptr<PointCloud> cloud,
        float voxel_size)
    {
        if (cloud->empty() || voxel_size <= 0.0f) {
            return cloud;
        }

        auto result = std::make_shared<PointCloud>();

        bool has_normals = cloud->hasNormals();
        bool has_colors  = cloud->hasColors();

        glm::vec3 min_bound(FLT_MAX);
        for (const auto& p : cloud->points) {
            min_bound = glm::min(min_bound, p);
        }
        glm::vec3 voxel_min_bound = min_bound - glm::vec3(voxel_size * 0.5f);

        struct VoxelData {
            glm::dvec3 sum_pos    = glm::dvec3(0.0);
            glm::dvec3 sum_normal = glm::dvec3(0.0);
            glm::dvec3 sum_color  = glm::dvec3(0.0);
            float min_bdist = 1e9f;
            int count = 0;
        };

        float inv_voxel = 1.0f / voxel_size;
        bool has_bdist = cloud->hasBoundaryDist();

        auto makeKey = [&voxel_min_bound, inv_voxel](const glm::vec3& p) -> std::tuple<int,int,int> {
            glm::vec3 ref = (p - voxel_min_bound) * inv_voxel;
            int ix = static_cast<int>(std::floor(ref.x));
            int iy = static_cast<int>(std::floor(ref.y));
            int iz = static_cast<int>(std::floor(ref.z));
            return {ix, iy, iz};
        };

        struct TupleHash {
            size_t operator()(const std::tuple<int,int,int>& t) const {
                auto h1 = std::hash<int>{}(std::get<0>(t));
                auto h2 = std::hash<int>{}(std::get<1>(t));
                auto h3 = std::hash<int>{}(std::get<2>(t));
                size_t seed = h1;
                seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                return seed;
            }
        };

        std::unordered_map<std::tuple<int,int,int>, VoxelData, TupleHash> voxelMap;

        for (size_t i = 0; i < cloud->size(); i++) {
            auto key = makeKey(cloud->points[i]);
            auto& vd = voxelMap[key];

            vd.sum_pos += glm::dvec3(cloud->points[i]);
            if (has_normals) vd.sum_normal += glm::dvec3(cloud->normals[i]);
            if (has_colors)  vd.sum_color  += glm::dvec3(cloud->colors[i]);
            if (has_bdist)   vd.min_bdist = std::min(vd.min_bdist, cloud->boundaryDist[i]);
            vd.count++;
        }

        result->reserve(voxelMap.size());

        for (const auto& [key, vd] : voxelMap) {
            double inv_count = 1.0 / vd.count;

            glm::vec3 avg_pos = glm::vec3(vd.sum_pos * inv_count);
            result->points.push_back(avg_pos);

            if (has_normals) {
                glm::vec3 avg_n = glm::vec3(vd.sum_normal * inv_count);
                result->normals.push_back(avg_n);
            }

            if (has_colors) {
                result->colors.push_back(glm::vec3(vd.sum_color * inv_count));
            }

            if (has_bdist) {
                result->boundaryDist.push_back(vd.min_bdist);
            }
        }

        std::cout << "  [Custom] VoxelDownSample: " << cloud->size()
                  << " -> " << result->size()
                  << " (voxel=" << voxel_size << ")" << std::endl;

        return result;
    }

    std::shared_ptr<PointCloud> preprocess(
        std::shared_ptr<PointCloud> cloud,
        float voxel_size,
        bool estimate_normals_flag = true)
    {
        std::cout << "  [Custom] Original points: " << cloud->size() << std::endl;

        auto cloud_down = voxelDownSample(cloud, voxel_size);

        std::cout << "  [Custom] Downsampled to: " << cloud_down->size() << " points" << std::endl;

        if (estimate_normals_flag || !cloud_down->hasNormals()) {
            estimateNormals(cloud_down, voxel_size * 2.0f, 30);
            std::cout << "  [Custom] Normals estimated" << std::endl;
        }

        return cloud_down;
    }

    void estimateNormals(
        std::shared_ptr<PointCloud> cloud,
        float search_radius,
        int max_nn)
    {
        if (cloud->empty()) return;

        size_t n = cloud->size();
        cloud->normals.resize(n, glm::vec3(0.0f, 0.0f, 1.0f));

        std::vector<glm::dvec3> ptsD(n);
        for (size_t i = 0; i < n; i++)
            ptsD[i] = glm::dvec3(cloud->points[i]);

        NanoflannAdaptorD adaptorD(ptsD);
        auto treeD = buildKDTreeD(adaptorD);

        double radius_sq = static_cast<double>(search_radius) * static_cast<double>(search_radius);

        for (size_t i = 0; i < n; i++) {
            std::vector<size_t> knn_indices;
            std::vector<double> knn_dists_sq;
            searchKNND(*treeD, ptsD[i], max_nn, knn_indices, knn_dists_sq);

            std::vector<size_t> neighbors;
            for (size_t j = 0; j < knn_indices.size(); j++) {
                if (knn_dists_sq[j] <= radius_sq) {
                    neighbors.push_back(knn_indices[j]);
                }
            }

            if (neighbors.size() < 3) {
                cloud->normals[i] = glm::vec3(0.0f, 0.0f, 1.0f);
                continue;
            }

            glm::dvec3 centroid(0.0);
            for (size_t idx : neighbors)
                centroid += ptsD[idx];
            centroid /= static_cast<double>(neighbors.size());

            double cov[3][3] = {};
            for (size_t idx : neighbors) {
                glm::dvec3 d = ptsD[idx] - centroid;
                cov[0][0] += d.x * d.x;
                cov[0][1] += d.x * d.y;
                cov[0][2] += d.x * d.z;
                cov[1][1] += d.y * d.y;
                cov[1][2] += d.y * d.z;
                cov[2][2] += d.z * d.z;
            }
            cov[1][0] = cov[0][1];
            cov[2][0] = cov[0][2];
            cov[2][1] = cov[1][2];

            glm::vec3 normal = computeSmallestEigenvector(cov);
            if (glm::length(normal) < 1e-8f) {
                normal = glm::vec3(0.0f, 0.0f, 1.0f);
            }
            cloud->normals[i] = normal;
        }

        orientNormalsConsistent(cloud, ptsD, *treeD, max_nn);

        std::cout << "  [Custom] estimateNormals: " << n << " normals computed"
                  << " (radius=" << search_radius << ", k=" << max_nn << ")" << std::endl;
    }

private:

    static glm::dvec3 computeEigenvector0(const double A[3][3], double eval0) {
        double row0[3] = {A[0][0] - eval0, A[0][1], A[0][2]};
        double row1[3] = {A[0][1], A[1][1] - eval0, A[1][2]};
        double row2[3] = {A[0][2], A[1][2], A[2][2] - eval0};

        glm::dvec3 r0(row0[0], row0[1], row0[2]);
        glm::dvec3 r1(row1[0], row1[1], row1[2]);
        glm::dvec3 r2(row2[0], row2[1], row2[2]);

        glm::dvec3 r0xr1 = glm::cross(r0, r1);
        glm::dvec3 r0xr2 = glm::cross(r0, r2);
        glm::dvec3 r1xr2 = glm::cross(r1, r2);

        double d0 = glm::dot(r0xr1, r0xr1);
        double d1 = glm::dot(r0xr2, r0xr2);
        double d2 = glm::dot(r1xr2, r1xr2);

        double dmax = d0;
        int imax = 0;
        if (d1 > dmax) { dmax = d1; imax = 1; }
        if (d2 > dmax) { imax = 2; }

        if (imax == 0) return r0xr1 / std::sqrt(d0);
        else if (imax == 1) return r0xr2 / std::sqrt(d1);
        else return r1xr2 / std::sqrt(d2);
    }

    static glm::dvec3 computeEigenvector1(const double A[3][3],
                                          const glm::dvec3& evec0, double eval1) {
        glm::dvec3 U, V;
        if (std::abs(evec0.x) > std::abs(evec0.y)) {
            double inv_length = 1.0 / std::sqrt(evec0.x * evec0.x + evec0.z * evec0.z);
            U = glm::dvec3(-evec0.z * inv_length, 0.0, evec0.x * inv_length);
        } else {
            double inv_length = 1.0 / std::sqrt(evec0.y * evec0.y + evec0.z * evec0.z);
            U = glm::dvec3(0.0, evec0.z * inv_length, -evec0.y * inv_length);
        }
        V = glm::cross(evec0, U);

        glm::dvec3 AU(
            A[0][0]*U.x + A[0][1]*U.y + A[0][2]*U.z,
            A[0][1]*U.x + A[1][1]*U.y + A[1][2]*U.z,
            A[0][2]*U.x + A[1][2]*U.y + A[2][2]*U.z);
        glm::dvec3 AV(
            A[0][0]*V.x + A[0][1]*V.y + A[0][2]*V.z,
            A[0][1]*V.x + A[1][1]*V.y + A[1][2]*V.z,
            A[0][2]*V.x + A[1][2]*V.y + A[2][2]*V.z);

        double m00 = glm::dot(U, AU) - eval1;
        double m01 = glm::dot(U, AV);
        double m11 = glm::dot(V, AV) - eval1;

        double absM00 = std::abs(m00);
        double absM01 = std::abs(m01);
        double absM11 = std::abs(m11);

        if (absM00 >= absM11) {
            double max_abs_comp = std::max(absM00, absM01);
            if (max_abs_comp > 0) {
                if (absM00 >= absM01) {
                    m01 /= m00;
                    m00 = 1.0 / std::sqrt(1.0 + m01 * m01);
                    m01 *= m00;
                } else {
                    m00 /= m01;
                    m01 = 1.0 / std::sqrt(1.0 + m00 * m00);
                    m00 *= m01;
                }
                return m01 * U - m00 * V;
            } else {
                return U;
            }
        } else {
            double max_abs_comp = std::max(absM11, absM01);
            if (max_abs_comp > 0) {
                if (absM11 >= absM01) {
                    m01 /= m11;
                    m11 = 1.0 / std::sqrt(1.0 + m01 * m01);
                    m01 *= m11;
                } else {
                    m11 /= m01;
                    m01 = 1.0 / std::sqrt(1.0 + m11 * m11);
                    m11 *= m01;
                }
                return m11 * U - m01 * V;
            } else {
                return U;
            }
        }
    }

    static glm::vec3 computeSmallestEigenvector(double cov[3][3]) {
        double max_coeff = 0.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                max_coeff = std::max(max_coeff, std::abs(cov[i][j]));

        if (max_coeff == 0.0) {
            return glm::vec3(0.0f, 0.0f, 0.0f);
        }

        double A[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                A[i][j] = cov[i][j] / max_coeff;

        double norm = A[0][1]*A[0][1] + A[0][2]*A[0][2] + A[1][2]*A[1][2];

        if (norm > 0) {
            double q = (A[0][0] + A[1][1] + A[2][2]) / 3.0;
            double b00 = A[0][0] - q;
            double b11 = A[1][1] - q;
            double b22 = A[2][2] - q;
            double p = std::sqrt((b00*b00 + b11*b11 + b22*b22 + norm*2.0) / 6.0);

            double c00 = b11*b22 - A[1][2]*A[1][2];
            double c01 = A[0][1]*b22 - A[1][2]*A[0][2];
            double c02 = A[0][1]*A[1][2] - b11*A[0][2];
            double det = (b00*c00 - A[0][1]*c01 + A[0][2]*c02) / (p*p*p);

            double half_det = std::min(std::max(det * 0.5, -1.0), 1.0);
            double angle = std::acos(half_det) / 3.0;
            double two_thirds_pi = 2.09439510239319549;
            double beta2 = std::cos(angle) * 2.0;
            double beta0 = std::cos(angle + two_thirds_pi) * 2.0;
            double beta1 = -(beta0 + beta2);

            double eval0 = q + p * beta0;
            double eval1 = q + p * beta1;
            double eval2 = q + p * beta2;

            glm::dvec3 result;

            if (half_det >= 0) {
                auto evec2 = computeEigenvector0(A, eval2);
                if (eval2 < eval0 && eval2 < eval1) {
                    return glm::vec3(evec2);
                }
                auto evec1 = computeEigenvector1(A, evec2, eval1);
                if (eval1 < eval0 && eval1 < eval2) {
                    return glm::vec3(evec1);
                }
                result = glm::cross(evec1, evec2);
            } else {
                auto evec0 = computeEigenvector0(A, eval0);
                if (eval0 < eval1 && eval0 < eval2) {
                    return glm::vec3(evec0);
                }
                auto evec1 = computeEigenvector1(A, evec0, eval1);
                if (eval1 < eval0 && eval1 < eval2) {
                    return glm::vec3(evec1);
                }
                result = glm::cross(evec0, evec1);
            }
            return glm::vec3(result);
        } else {
            if (A[0][0] < A[1][1] && A[0][0] < A[2][2]) {
                return glm::vec3(1.0f, 0.0f, 0.0f);
            } else if (A[1][1] < A[0][0] && A[1][1] < A[2][2]) {
                return glm::vec3(0.0f, 1.0f, 0.0f);
            } else {
                return glm::vec3(0.0f, 0.0f, 1.0f);
            }
        }
    }

    static void orientNormalsConsistent(
        std::shared_ptr<PointCloud> cloud,
        const std::vector<glm::dvec3>& ptsD,
        const KDTree3DD& tree,
        int k)
    {
        size_t n = cloud->size();
        if (n == 0) return;

        struct WeightedEdge {
            size_t v0, v1;
            double weight;
            bool operator<(const WeightedEdge& other) const { return weight < other.weight; }
        };

        struct DisjointSet {
            std::vector<size_t> parent_, size_;
            DisjointSet(size_t sz) : parent_(sz), size_(sz, 0) {
                std::iota(parent_.begin(), parent_.end(), 0);
            }
            size_t find(size_t x) {
                if (parent_[x] != x) parent_[x] = find(parent_[x]);
                return parent_[x];
            }
            void unite(size_t x, size_t y) {
                x = find(x); y = find(y);
                if (x != y) {
                    if (size_[x] < size_[y]) { size_[y] += size_[x]; parent_[x] = y; }
                    else { size_[x] += size_[y]; parent_[y] = x; }
                }
            }
        };

        auto kruskal = [](std::vector<WeightedEdge>& edges, size_t nv) -> std::vector<WeightedEdge> {
            std::sort(edges.begin(), edges.end());
            DisjointSet ds(nv);
            std::vector<WeightedEdge> mst;
            mst.reserve(nv - 1);
            for (auto& e : edges) {
                if (ds.find(e.v0) != ds.find(e.v1)) {
                    mst.push_back(e);
                    ds.unite(e.v0, e.v1);
                }
            }
            return mst;
        };

        auto edgeKey = [n](size_t v0, size_t v1) -> size_t {
            return std::min(v0, v1) * n + std::max(v0, v1);
        };
        std::unordered_set<size_t> graphEdges;

        auto normalWeight = [&](size_t v0, size_t v1) -> double {
            glm::dvec3 n0(cloud->normals[v0]);
            glm::dvec3 n1(cloud->normals[v1]);
            return 1.0 - std::abs(glm::dot(n0, n1));
        };

        std::vector<WeightedEdge> euclideanEdges;

        if (n < 2000) {
            euclideanEdges.reserve(n * (n - 1) / 2);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = i + 1; j < n; j++) {
                    glm::dvec3 diff = ptsD[i] - ptsD[j];
                    double dist = glm::dot(diff, diff);
                    euclideanEdges.push_back({i, j, dist});
                    graphEdges.insert(edgeKey(i, j));
                }
            }
        } else {
            int k_euclidean = std::min(static_cast<int>(n - 1), k * 3);
            euclideanEdges.reserve(n * k_euclidean);
            for (size_t i = 0; i < n; i++) {
                std::vector<size_t> indices;
                std::vector<double> dists_sq;
                searchKNND(tree, ptsD[i], k_euclidean, indices, dists_sq);
                for (size_t j = 0; j < indices.size(); j++) {
                    size_t idx = indices[j];
                    if (idx == i) continue;
                    size_t key = edgeKey(i, idx);
                    if (graphEdges.count(key) == 0) {
                        euclideanEdges.push_back({i, idx, dists_sq[j]});
                        graphEdges.insert(key);
                    }
                }
            }
        }

        auto mstEdges = kruskal(euclideanEdges, n);

        graphEdges.clear();

        for (auto& e : mstEdges) {
            e.weight = normalWeight(e.v0, e.v1);
            graphEdges.insert(edgeKey(e.v0, e.v1));
        }

        for (size_t i = 0; i < n; i++) {
            std::vector<size_t> indices;
            std::vector<double> dists_sq;
            searchKNND(tree, ptsD[i], k, indices, dists_sq);

            for (size_t j = 0; j < indices.size(); j++) {
                size_t idx = indices[j];
                if (idx == i) continue;
                size_t key = edgeKey(i, idx);
                if (graphEdges.count(key) == 0) {
                    double w = normalWeight(i, idx);
                    mstEdges.push_back({i, idx, w});
                    graphEdges.insert(key);
                }
            }
        }

        auto riemannianMST = kruskal(mstEdges, n);

        std::vector<std::vector<size_t>> mstAdj(n);
        for (const auto& e : riemannianMST) {
            mstAdj[e.v0].push_back(e.v1);
            mstAdj[e.v1].push_back(e.v0);
        }

        double minZ = std::numeric_limits<double>::max();
        size_t startIdx = 0;
        for (size_t i = 0; i < n; i++) {
            double z = ptsD[i].z;
            if (z < minZ) {
                minZ = z;
                startIdx = i;
            }
        }

        {
            glm::dvec3 ns(cloud->normals[startIdx]);
            if (glm::dot(glm::dvec3(0.0, 0.0, -1.0), ns) < 0.0) {
                cloud->normals[startIdx] = -cloud->normals[startIdx];
            }
        }

        std::vector<bool> visited(n, false);
        std::queue<size_t> bfsQueue;
        bfsQueue.push(startIdx);

        while (!bfsQueue.empty()) {
            size_t cur = bfsQueue.front();
            bfsQueue.pop();
            visited[cur] = true;

            for (size_t neighbor : mstAdj[cur]) {
                if (visited[neighbor]) continue;

                glm::dvec3 n0(cloud->normals[cur]);
                glm::dvec3 n1(cloud->normals[neighbor]);
                if (glm::dot(n0, n1) < 0.0) {
                    cloud->normals[neighbor] = -cloud->normals[neighbor];
                }

                bfsQueue.push(neighbor);
            }
        }

        for (size_t i = 0; i < n; i++) {
            if (!visited[i]) {
                if (cloud->normals[i].z > 0.0f) {
                    cloud->normals[i] = -cloud->normals[i];
                }
            }
        }
    }

    static bool solve6x6(double A[6][6], double b[6], double x[6]) {
        const int N = 6;

        double L[N][N] = {};
        double D[N] = {};

        for (int j = 0; j < N; j++) {
            double sum = A[j][j];
            for (int k = 0; k < j; k++) {
                sum -= L[j][k] * L[j][k] * D[k];
            }
            D[j] = sum;

            L[j][j] = 1.0;
            if (std::abs(D[j]) < 1e-20) {
                D[j] = 0.0;
                for (int i = j + 1; i < N; i++) {
                    L[i][j] = 0.0;
                }
            } else {
                for (int i = j + 1; i < N; i++) {
                    double s = A[i][j];
                    for (int k = 0; k < j; k++) {
                        s -= L[i][k] * L[j][k] * D[k];
                    }
                    L[i][j] = s / D[j];
                }
            }
        }

        double y[N];
        for (int i = 0; i < N; i++) {
            double s = b[i];
            for (int k = 0; k < i; k++) {
                s -= L[i][k] * y[k];
            }
            y[i] = s;
        }

        double z[N];
        for (int i = 0; i < N; i++) {
            if (std::abs(D[i]) < 1e-20) {
                z[i] = 0.0;
            } else {
                z[i] = y[i] / D[i];
            }
        }

        for (int i = N - 1; i >= 0; i--) {
            double s = z[i];
            for (int k = i + 1; k < N; k++) {
                s -= L[k][i] * x[k];
            }
            x[i] = s;
        }

        return true;
    }

    static glm::dmat4 transformVector6dToMatrix4d(const double result[6]) {
        double alpha = result[0];
        double beta  = result[1];
        double gamma = result[2];

        double ca = std::cos(alpha), sa = std::sin(alpha);
        double cb = std::cos(beta),  sb = std::sin(beta);
        double cg = std::cos(gamma), sg = std::sin(gamma);

        glm::dmat4 mat(1.0);
        mat[0][0] =  cg*cb;
        mat[0][1] =  sg*cb;
        mat[0][2] = -sb;

        mat[1][0] =  cg*sb*sa - sg*ca;
        mat[1][1] =  sg*sb*sa + cg*ca;
        mat[1][2] =  cb*sa;

        mat[2][0] =  cg*sb*ca + sg*sa;
        mat[2][1] =  sg*sb*ca - cg*sa;
        mat[2][2] =  cb*ca;

        mat[3][0] = result[3];
        mat[3][1] = result[4];
        mat[3][2] = result[5];
        mat[3][3] = 1.0;

        return mat;
    }

    static std::vector<std::pair<int, int>> initialMatching(
        const FeatureSet& src_features,
        const FeatureSet& dst_features)
    {
        int src_n = static_cast<int>(src_features.numPoints());
        int dst_n = static_cast<int>(dst_features.numPoints());

        if (src_n == 0 || dst_n == 0) return {};

        NanoflannFeatureAdaptor src_adaptor(src_features.data, src_features.dimension);
        NanoflannFeatureAdaptor dst_adaptor(dst_features.data, dst_features.dimension);
        auto src_tree = buildFeatureKDTree(src_adaptor);
        auto dst_tree = buildFeatureKDTree(dst_adaptor);

        std::map<int, int> corres_ij;
        std::vector<int> corres_ji(dst_n, -1);

        for (int j = 0; j < dst_n; j++) {
            const size_t K = 1;
            size_t nn_idx[K];
            float nn_dist[K];
            nanoflann::KNNResultSet<float> resultSet(K);
            resultSet.init(nn_idx, nn_dist);
            src_tree->findNeighbors(resultSet, dst_features.data[j].data());

            int i = static_cast<int>(nn_idx[0]);
            corres_ji[j] = i;

            if (corres_ij.find(i) == corres_ij.end()) {
                corres_ij[i] = -1;
                size_t nn_idx2[K];
                float nn_dist2[K];
                nanoflann::KNNResultSet<float> resultSet2(K);
                resultSet2.init(nn_idx2, nn_dist2);
                dst_tree->findNeighbors(resultSet2, src_features.data[i].data());
                corres_ij[i] = static_cast<int>(nn_idx2[0]);
            }
        }

        std::vector<std::pair<int, int>> corres_cross;
        for (const auto& [i, j] : corres_ij) {
            if (j >= 0 && j < dst_n && corres_ji[j] == i) {
                corres_cross.push_back({i, j});
            }
        }

        std::cout << "    [FGR] Initial matching: " << corres_cross.size()
                  << " correspondences (cross-check)" << std::endl;

        return corres_cross;
    }

    static std::vector<std::pair<int, int>> advancedMatching(
        const PointCloud& src_cloud,
        const PointCloud& dst_cloud,
        const std::vector<std::pair<int, int>>& corres_cross,
        double tuple_scale,
        int maximum_tuple_count)
    {
        int ncorr = static_cast<int>(corres_cross.size());
        if (ncorr < 3) return corres_cross;

        int number_of_trial = ncorr * 100;
        int cnt = 0;

        static std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, ncorr - 1);

        std::vector<std::pair<int, int>> corres_tuple;

        for (int trial = 0; trial < number_of_trial; trial++) {
            int rand0 = dist(gen);
            int rand1 = dist(gen);
            int rand2 = dist(gen);

            int idi0 = corres_cross[rand0].first;
            int idj0 = corres_cross[rand0].second;
            int idi1 = corres_cross[rand1].first;
            int idj1 = corres_cross[rand1].second;
            int idi2 = corres_cross[rand2].first;
            int idj2 = corres_cross[rand2].second;

            double li0 = glm::length(glm::dvec3(src_cloud.points[idi0]) -
                                     glm::dvec3(src_cloud.points[idi1]));
            double li1 = glm::length(glm::dvec3(src_cloud.points[idi1]) -
                                     glm::dvec3(src_cloud.points[idi2]));
            double li2 = glm::length(glm::dvec3(src_cloud.points[idi2]) -
                                     glm::dvec3(src_cloud.points[idi0]));

            double lj0 = glm::length(glm::dvec3(dst_cloud.points[idj0]) -
                                     glm::dvec3(dst_cloud.points[idj1]));
            double lj1 = glm::length(glm::dvec3(dst_cloud.points[idj1]) -
                                     glm::dvec3(dst_cloud.points[idj2]));
            double lj2 = glm::length(glm::dvec3(dst_cloud.points[idj2]) -
                                     glm::dvec3(dst_cloud.points[idj0]));

            if ((li0 * tuple_scale < lj0) && (lj0 < li0 / tuple_scale) &&
                (li1 * tuple_scale < lj1) && (lj1 < li1 / tuple_scale) &&
                (li2 * tuple_scale < lj2) && (lj2 < li2 / tuple_scale)) {
                corres_tuple.push_back({idi0, idj0});
                corres_tuple.push_back({idi1, idj1});
                corres_tuple.push_back({idi2, idj2});
                cnt++;
            }

            if (cnt >= maximum_tuple_count) break;
        }

        std::cout << "    [FGR] Tuple test: " << cnt << " tuples, "
                  << corres_tuple.size() << " correspondences" << std::endl;

        return corres_tuple;
    }

    struct NormalizeResult {
        glm::dvec3 mean_source;
        glm::dvec3 mean_target;
        double scale_global;
        double scale_start;
    };

    static NormalizeResult normalizePointClouds(
        std::vector<glm::dvec3>& source_pts,
        std::vector<glm::dvec3>& target_pts,
        bool use_absolute_scale)
    {
        NormalizeResult result;
        double scale = 0.0;

        result.mean_source = glm::dvec3(0.0);
        for (const auto& p : source_pts) result.mean_source += p;
        result.mean_source /= static_cast<double>(source_pts.size());

        result.mean_target = glm::dvec3(0.0);
        for (const auto& p : target_pts) result.mean_target += p;
        result.mean_target /= static_cast<double>(target_pts.size());

        for (auto& p : source_pts) p -= result.mean_source;
        for (auto& p : target_pts) p -= result.mean_target;

        for (const auto& p : source_pts) {
            double d = glm::length(p);
            if (d > scale) scale = d;
        }
        for (const auto& p : target_pts) {
            double d = glm::length(p);
            if (d > scale) scale = d;
        }

        if (use_absolute_scale) {
            result.scale_global = 1.0;
            result.scale_start = scale;
        } else {
            result.scale_global = scale;
            result.scale_start = 1.0;
        }

        if (result.scale_global > 0.0) {
            for (auto& p : source_pts) p /= result.scale_global;
            for (auto& p : target_pts) p /= result.scale_global;
        }

        std::cout << "    [FGR] Normalize: scale_global=" << result.scale_global
                  << ", scale_start=" << result.scale_start
                  << " (par init=" << result.scale_global << ")" << std::endl;

        return result;
    }

    static glm::dmat4 optimizePairwiseRegistration(
        const std::vector<glm::dvec3>& source_pts,
        std::vector<glm::dvec3> target_pts_copy,
        const std::vector<std::pair<int, int>>& corres,
        double scale_start,
        double division_factor,
        bool decrease_mu,
        double maximum_correspondence_distance,
        int iteration_number)
    {
        if (corres.size() < 10) {
            std::cout << "    [FGR] Too few correspondences (" << corres.size()
            << " < 10), returning identity (Open3D behavior)" << std::endl;
            return glm::dmat4(1.0);
        }

        double par = scale_start;
        std::vector<double> s(corres.size(), 1.0);
        glm::dmat4 trans(1.0);

        for (int itr = 0; itr < iteration_number; itr++) {
            double JTJ[6][6] = {};
            double JTr_vec[6] = {};
            double J[6];

            for (size_t c = 0; c < corres.size(); c++) {
                int ii = corres[c].first;
                int jj = corres[c].second;

                glm::dvec3 p = source_pts[ii];
                glm::dvec3 q = target_pts_copy[jj];
                glm::dvec3 rpq = p - q;

                double rpq_dot = glm::dot(rpq, rpq);
                double temp = par / (rpq_dot + par);
                s[c] = temp * temp;

                std::fill(J, J + 6, 0.0);
                J[1] = -q.z;
                J[2] = q.y;
                J[3] = -1.0;
                double r = rpq.x;
                for (int a = 0; a < 6; a++) {
                    JTr_vec[a] += J[a] * r * s[c];
                    for (int b = 0; b < 6; b++)
                        JTJ[a][b] += J[a] * J[b] * s[c];
                }

                std::fill(J, J + 6, 0.0);
                J[2] = -q.x;
                J[0] = q.z;
                J[4] = -1.0;
                r = rpq.y;
                for (int a = 0; a < 6; a++) {
                    JTr_vec[a] += J[a] * r * s[c];
                    for (int b = 0; b < 6; b++)
                        JTJ[a][b] += J[a] * J[b] * s[c];
                }

                std::fill(J, J + 6, 0.0);
                J[0] = -q.y;
                J[1] = q.x;
                J[5] = -1.0;
                r = rpq.z;
                for (int a = 0; a < 6; a++) {
                    JTr_vec[a] += J[a] * r * s[c];
                    for (int b = 0; b < 6; b++)
                        JTJ[a][b] += J[a] * J[b] * s[c];
                }
            }

            double negJTJ[6][6];
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                    negJTJ[a][b] = -JTJ[a][b];

            double delta_x[6] = {};
            if (!solve6x6(negJTJ, JTr_vec, delta_x)) {
                break;
            }

            glm::dmat4 delta = transformVector6dToMatrix4d(delta_x);
            trans = delta * trans;

            for (size_t idx = 0; idx < target_pts_copy.size(); idx++) {
                glm::dvec4 pt(target_pts_copy[idx], 1.0);
                glm::dvec4 transformed = delta * pt;
                target_pts_copy[idx] = glm::dvec3(transformed);
            }

            if (decrease_mu) {
                if (itr % 4 == 0 && par > maximum_correspondence_distance) {
                    par /= division_factor;
                }
            }
        }

        return trans;
    }

    static glm::dmat4 getTransformationOriginalScale(
        const glm::dmat4& transformation,
        const glm::dvec3& mean_source,
        const glm::dvec3& mean_target,
        double scale_global)
    {
        glm::dmat3 R;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[i][j] = transformation[i][j];

        glm::dvec3 t(transformation[3][0], transformation[3][1], transformation[3][2]);

        glm::dmat4 result(0.0);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                result[i][j] = R[i][j];

        glm::dvec3 trans_t = -R * mean_target + t * scale_global + mean_source;
        result[3][0] = trans_t.x;
        result[3][1] = trans_t.y;
        result[3][2] = trans_t.z;
        result[3][3] = 1.0;

        return result;
    }

    static RegistrationResult evaluateRegistration(
        const PointCloud& source,
        const PointCloud& target,
        double max_correspondence_distance,
        const glm::dmat4& transformation)
    {
        RegistrationResult result;
        result.transformation = glm::mat4(transformation);

        if (source.empty() || target.empty()) return result;

        NanoflannAdaptor adaptor(target.points);
        auto tree = buildKDTree(adaptor);

        double max_dist_sq = max_correspondence_distance * max_correspondence_distance;
        int inlier_count = 0;
        double error_sum = 0.0;

        for (size_t i = 0; i < source.size(); i++) {
            glm::dvec4 pt(source.points[i], 1.0);
            glm::dvec4 transformed = transformation * pt;
            glm::vec3 tp(static_cast<float>(transformed.x),
                         static_cast<float>(transformed.y),
                         static_cast<float>(transformed.z));

            size_t nn_idx;
            float nn_dist_sq;
            searchKNN1(*tree, tp, nn_idx, nn_dist_sq);

            if (static_cast<double>(nn_dist_sq) < max_dist_sq) {
                inlier_count++;
                error_sum += static_cast<double>(nn_dist_sq);
                result.correspondences.push_back(
                    {static_cast<int>(i), static_cast<int>(nn_idx)});
            }
        }

        result.fitness = (source.size() > 0) ?
                             static_cast<float>(inlier_count) / static_cast<float>(source.size()) : 0.0f;
        result.inlier_rmse = (inlier_count > 0) ?
                                 static_cast<float>(std::sqrt(error_sum / inlier_count)) : 0.0f;

        return result;
    }

public:

    static RegistrationResult evaluateCurrentFitness(
        const PointCloud& source,
        const PointCloud& target,
        double max_correspondence_distance,
        const glm::dmat4& transformation = glm::dmat4(1.0))
    {
        return evaluateRegistration(source, target, max_correspondence_distance, transformation);
    }

    std::shared_ptr<FeatureSet> computeFPFH(
        std::shared_ptr<PointCloud> cloud,
        float voxel_size)
    {
        auto features = std::make_shared<FeatureSet>();
        size_t n = cloud->size();

        if (n == 0 || !cloud->hasNormals()) {
            std::cout << "  [Custom] computeFPFH: empty cloud or no normals" << std::endl;
            features->resize(33, 0);
            return features;
        }

        features->resize(33, static_cast<int>(n));

        float search_radius = voxel_size * 5.0f;
        int max_nn = 100;

        NanoflannAdaptor adaptor(cloud->points);
        auto tree = buildKDTree(adaptor);

        float radius_sq = search_radius * search_radius;

        std::vector<std::vector<float>> spfh(n, std::vector<float>(33, 0.0f));
        std::vector<std::vector<size_t>> neighborCache(n);
        std::vector<std::vector<float>> dist2Cache(n);

        for (size_t i = 0; i < n; i++) {
            std::vector<size_t> knn_indices;
            std::vector<float> knn_dists_sq;
            searchKNN(*tree, cloud->points[i], max_nn, knn_indices, knn_dists_sq);

            std::vector<size_t> neighbors;
            std::vector<float> neighbor_dists2;

            for (size_t j = 0; j < knn_indices.size(); j++) {
                if (knn_indices[j] == i) continue;
                if (knn_dists_sq[j] <= radius_sq) {
                    neighbors.push_back(knn_indices[j]);
                    neighbor_dists2.push_back(knn_dists_sq[j]);
                }
            }

            neighborCache[i] = neighbors;
            dist2Cache[i] = neighbor_dists2;

            if (neighbors.empty()) continue;

            double hist_incr = 100.0 / static_cast<double>(neighbors.size());

            const glm::vec3& pi = cloud->points[i];
            const glm::vec3& ni = cloud->normals[i];

            for (size_t j = 0; j < neighbors.size(); j++) {
                size_t idx = neighbors[j];
                const glm::vec3& pj = cloud->points[idx];
                const glm::vec3& nj = cloud->normals[idx];

                glm::vec3 dp2p1 = pj - pi;
                float dp_len = glm::length(dp2p1);
                if (dp_len == 0.0f) continue;

                float angle1 = glm::dot(ni, dp2p1) / dp_len;
                float angle2 = glm::dot(nj, dp2p1) / dp_len;

                glm::vec3 n1_copy = ni;
                glm::vec3 n2_copy = nj;
                float phi_val;

                if (std::abs(angle1) < std::abs(angle2)) {
                    n1_copy = nj;
                    n2_copy = ni;
                    dp2p1 *= -1.0f;
                    phi_val = -angle2;
                } else {
                    phi_val = angle1;
                }

                glm::vec3 dp_norm = dp2p1 / dp_len;
                glm::vec3 v = glm::cross(dp_norm, n1_copy);
                float v_len = glm::length(v);
                if (v_len == 0.0f) continue;
                v /= v_len;

                glm::vec3 w = glm::cross(n1_copy, v);

                float theta_val = std::atan2(glm::dot(w, n2_copy),
                                             glm::dot(n1_copy, n2_copy));
                float alpha_val = glm::dot(v, n2_copy);

                int bin_theta = static_cast<int>(std::floor(
                    11.0f * (theta_val + static_cast<float>(M_PI)) /
                    (2.0f * static_cast<float>(M_PI))));
                bin_theta = std::max(0, std::min(10, bin_theta));

                int bin_alpha = static_cast<int>(std::floor(
                    11.0f * (alpha_val + 1.0f) * 0.5f));
                bin_alpha = std::max(0, std::min(10, bin_alpha));

                int bin_phi = static_cast<int>(std::floor(
                    11.0f * (phi_val + 1.0f) * 0.5f));
                bin_phi = std::max(0, std::min(10, bin_phi));

                spfh[i][bin_theta]       += static_cast<float>(hist_incr);
                spfh[i][11 + bin_alpha]  += static_cast<float>(hist_incr);
                spfh[i][22 + bin_phi]    += static_cast<float>(hist_incr);
            }
        }

        for (size_t i = 0; i < n; i++) {
            const auto& neighbors = neighborCache[i];
            const auto& dists2 = dist2Cache[i];

            if (neighbors.empty()) {
                features->data[i] = spfh[i];
                continue;
            }

            double sum[3] = {0.0, 0.0, 0.0};
            std::vector<double> feat_accum(33, 0.0);

            for (size_t j = 0; j < neighbors.size(); j++) {
                double dist2 = static_cast<double>(dists2[j]);
                if (dist2 == 0.0) continue;

                for (int d = 0; d < 33; d++) {
                    double val = static_cast<double>(spfh[neighbors[j]][d]) / dist2;
                    sum[d / 11] += val;
                    feat_accum[d] += val;
                }
            }

            for (int j = 0; j < 3; j++) {
                if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
            }

            for (int d = 0; d < 33; d++) {
                features->data[i][d] = static_cast<float>(
                    feat_accum[d] * sum[d / 11] + static_cast<double>(spfh[i][d])
                    );
            }
        }

        std::cout << "  [Custom] computeFPFH: " << n << " points, dim=" << features->dimension
                  << " (radius=" << search_radius << ", max_nn=" << max_nn << ")" << std::endl;

        return features;
    }

    RegistrationResult fastGlobalRegistration(
        std::shared_ptr<PointCloud> source,
        std::shared_ptr<PointCloud> target,
        std::shared_ptr<FeatureSet> source_fpfh,
        std::shared_ptr<FeatureSet> target_fpfh,
        float voxel_size)
    {
        float distance_threshold = voxel_size * 0.5f;

        double division_factor = 1.4;
        bool use_absolute_scale = true;
        bool decrease_mu = true;
        double maximum_correspondence_distance = static_cast<double>(distance_threshold);
        int iteration_number = 64;
        double tuple_scale = 0.95;
        int maximum_tuple_count = 1000;

        std::cout << "  [Custom FGR] Distance threshold: " << distance_threshold << std::endl;
        std::cout << "  [Custom FGR] Source: " << source->size()
                  << " pts, Target: " << target->size() << " pts" << std::endl;

        auto fgr_start = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<int, int>> corres;

        if (source->size() >= target->size()) {
            auto cross = initialMatching(*source_fpfh, *target_fpfh);
            corres = advancedMatching(*source, *target, cross,
                                      tuple_scale, maximum_tuple_count);
        } else {
            auto cross = initialMatching(*target_fpfh, *source_fpfh);
            corres = advancedMatching(*target, *source, cross,
                                      tuple_scale, maximum_tuple_count);
            for (auto& p : corres) std::swap(p.first, p.second);
        }

        std::vector<glm::dvec3> source_norm(source->size());
        std::vector<glm::dvec3> target_norm(target->size());
        for (size_t i = 0; i < source->size(); i++)
            source_norm[i] = glm::dvec3(source->points[i]);
        for (size_t i = 0; i < target->size(); i++)
            target_norm[i] = glm::dvec3(target->points[i]);

        auto normResult = normalizePointClouds(source_norm, target_norm,
                                               use_absolute_scale);

        glm::dmat4 trans_normalized = optimizePairwiseRegistration(
            source_norm,
            target_norm,
            corres,
            normResult.scale_global,
            division_factor,
            decrease_mu,
            maximum_correspondence_distance,
            iteration_number
            );

        glm::dmat4 trans_original = getTransformationOriginalScale(
            trans_normalized,
            normResult.mean_source,
            normResult.mean_target,
            normResult.scale_global
            );

        glm::dmat4 final_transform = glm::inverse(trans_original);

        auto result = evaluateRegistration(
            *source, *target,
            maximum_correspondence_distance,
            final_transform
            );

        auto fgr_end = std::chrono::high_resolution_clock::now();
        auto fgr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          fgr_end - fgr_start).count();

        std::cout << "  [Custom FGR] Fitness: " << result.fitness << std::endl;
        std::cout << "  [Custom FGR] RMSE:    " << result.inlier_rmse << std::endl;
        std::cout << "  [Custom FGR] Time:    " << fgr_ms << "ms" << std::endl;

        std::cout << "  [Custom FGR] Transformation:" << std::endl;
        const auto& T = result.transformation;
        for (int row = 0; row < 4; row++) {
            std::cout << "    [";
            for (int col = 0; col < 4; col++) {
                std::cout << std::fixed << std::setprecision(6) << T[col][row];
                if (col < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        return result;
    }

    RegistrationResult icpRefinement(
        std::shared_ptr<PointCloud> source,
        std::shared_ptr<PointCloud> target,
        const glm::mat4& init_transformation,
        float distance_threshold,
        bool point_to_plane = true,
        double convergence_fitness = 1e-6,
        double convergence_rmse = 1e-6,
        int max_iteration = 30,
        bool verbose = false)
    {
        if (source->empty() || target->empty()) {
            RegistrationResult result;
            result.transformation = init_transformation;
            return result;
        }

        if (point_to_plane && !target->hasNormals()) {
            std::cout << "  [Custom ICP] Warning: target has no normals, "
                         "falling back to point-to-point" << std::endl;
            point_to_plane = false;
        }

        std::vector<glm::dvec3> target_pts_d(target->size());
        for (size_t i = 0; i < target->size(); i++) {
            target_pts_d[i] = glm::dvec3(target->points[i]);
        }
        NanoflannAdaptorD adaptorD(target_pts_d);
        auto tree = buildKDTreeD(adaptorD);

        glm::dmat4 transformation = glm::dmat4(init_transformation);
        double max_dist_sq = static_cast<double>(distance_threshold) * distance_threshold;

        std::vector<glm::dvec3> pcd(source->size());
        for (size_t i = 0; i < source->size(); i++) {
            glm::dvec4 pt(source->points[i], 1.0);
            glm::dvec4 tp = transformation * pt;
            pcd[i] = glm::dvec3(tp);
        }

        struct CorrespondenceResult {
            double fitness;
            double rmse;
            std::vector<std::pair<int,int>> corres;
        };

        auto findCorrespondences = [&](const std::vector<glm::dvec3>& src_pts)
            -> CorrespondenceResult
        {
            CorrespondenceResult cr;
            double error2 = 0.0;

            for (int i = 0; i < static_cast<int>(src_pts.size()); i++) {
                size_t nn_idx;
                double nn_dist_sq;
                searchKNN1D(*tree, src_pts[i], nn_idx, nn_dist_sq);

                if (nn_dist_sq < max_dist_sq) {
                    cr.corres.push_back({i, static_cast<int>(nn_idx)});
                    error2 += nn_dist_sq;
                }
            }

            cr.fitness = src_pts.empty() ? 0.0 :
                             static_cast<double>(cr.corres.size()) /
                                 static_cast<double>(src_pts.size());
            cr.rmse = cr.corres.empty() ? 0.0 :
                          std::sqrt(error2 / static_cast<double>(cr.corres.size()));
            return cr;
        };

        auto computeTransformation = [&](
                                         const std::vector<glm::dvec3>& src_pts,
                                         const std::vector<std::pair<int,int>>& corr_set,
                                         bool p2plane) -> glm::dmat4
        {
            double JTJ[6][6] = {};
            double JTr_vec[6] = {};

            if (p2plane) {
                for (const auto& c : corr_set) {
                    glm::dvec3 vs = src_pts[c.first];
                    glm::dvec3 vt = glm::dvec3(target->points[c.second]);
                    glm::dvec3 nt = glm::dvec3(target->normals[c.second]);

                    double r = glm::dot(vs - vt, nt);

                    double J[6];
                    J[0] = vs.y * nt.z - vs.z * nt.y;
                    J[1] = vs.z * nt.x - vs.x * nt.z;
                    J[2] = vs.x * nt.y - vs.y * nt.x;
                    J[3] = nt.x;
                    J[4] = nt.y;
                    J[5] = nt.z;

                    for (int a = 0; a < 6; a++) {
                        JTr_vec[a] += J[a] * r;
                        for (int b = 0; b < 6; b++)
                            JTJ[a][b] += J[a] * J[b];
                    }
                }
            } else {
                for (const auto& c : corr_set) {
                    glm::dvec3 vs = src_pts[c.first];
                    glm::dvec3 vt = glm::dvec3(target->points[c.second]);
                    glm::dvec3 rpq = vs - vt;

                    double J[6];
                    std::fill(J, J + 6, 0.0);
                    J[1] = -vs.z; J[2] = vs.y; J[3] = -1.0;
                    double r = rpq.x;
                    for (int a = 0; a < 6; a++) {
                        JTr_vec[a] += J[a] * r;
                        for (int b = 0; b < 6; b++)
                            JTJ[a][b] += J[a] * J[b];
                    }

                    std::fill(J, J + 6, 0.0);
                    J[0] = vs.z; J[2] = -vs.x; J[4] = -1.0;
                    r = rpq.y;
                    for (int a = 0; a < 6; a++) {
                        JTr_vec[a] += J[a] * r;
                        for (int b = 0; b < 6; b++)
                            JTJ[a][b] += J[a] * J[b];
                    }

                    std::fill(J, J + 6, 0.0);
                    J[0] = -vs.y; J[1] = vs.x; J[5] = -1.0;
                    r = rpq.z;
                    for (int a = 0; a < 6; a++) {
                        JTr_vec[a] += J[a] * r;
                        for (int b = 0; b < 6; b++)
                            JTJ[a][b] += J[a] * J[b];
                    }
                }
            }

            double neg_JTr[6];
            for (int a = 0; a < 6; a++) neg_JTr[a] = -JTr_vec[a];

            double x[6] = {};
            if (!solve6x6(JTJ, neg_JTr, x)) {
                return glm::dmat4(1.0);
            }

            return transformVector6dToMatrix4d(x);
        };

        auto eval = findCorrespondences(pcd);
        if (verbose) {
            std::cout << "    [ICP init] corres=" << eval.corres.size()
            << " fitness=" << std::fixed << std::setprecision(6)
            << eval.fitness << " rmse=" << eval.rmse << std::endl;
        }

        for (int i = 0; i < max_iteration; i++) {
            if (eval.corres.size() < 3) break;

            glm::dmat4 update = computeTransformation(
                pcd, eval.corres, point_to_plane);

            transformation = update * transformation;
            for (auto& pt : pcd) {
                glm::dvec4 p4(pt, 1.0);
                glm::dvec4 tp = update * p4;
                pt = glm::dvec3(tp);
            }

            auto backup = eval;
            eval = findCorrespondences(pcd);

            if (verbose) {
                std::cout << "    [ICP iter " << i << "] "
                          << "corres=" << eval.corres.size()
                          << " fitness=" << std::fixed << std::setprecision(6)
                          << eval.fitness
                          << " rmse=" << eval.rmse
                          << " df=" << std::abs(backup.fitness - eval.fitness)
                          << " dr=" << std::abs(backup.rmse - eval.rmse)
                          << std::endl;
            }

            if (std::abs(backup.fitness - eval.fitness) < convergence_fitness &&
                std::abs(backup.rmse - eval.rmse) < convergence_rmse) {
                if (verbose) {
                    std::cout << "    [ICP] Converged at iteration " << i << std::endl;
                }
                break;
            }
        }

        RegistrationResult result;
        result.fitness = static_cast<float>(eval.fitness);
        result.inlier_rmse = static_cast<float>(eval.rmse);
        result.transformation = glm::mat4(transformation);
        for (const auto& c : eval.corres) {
            result.correspondences.push_back(c);
        }
        return result;
    }

    void extractCorrespondences(
        std::shared_ptr<PointCloud> source,
        std::shared_ptr<PointCloud> target,
        const glm::mat4& transformation,
        float max_distance,
        std::vector<glm::vec3>& source_points,
        std::vector<glm::vec3>& target_points)
    {
        source_points.clear();
        target_points.clear();

        if (!source || !target || source->empty() || target->empty()) return;

        NanoflannAdaptor targetAdaptor(target->points);
        auto tree = buildKDTree(targetAdaptor);

        float max_dist_sq = max_distance * max_distance;

        for (size_t i = 0; i < source->size(); i++) {
            glm::vec4 transformed = transformation * glm::vec4(source->points[i], 1.0f);
            glm::vec3 pt_transformed(transformed);

            size_t idx;
            float dist_sq;
            if (searchKNN1(*tree, pt_transformed, idx, dist_sq)) {
                if (dist_sq < max_dist_sq) {
                    source_points.push_back(source->points[i]);
                    target_points.push_back(target->points[idx]);
                }
            }
        }
    }

    Reg3D::UmeyamaResult applyUmeyamaRefinement(
        std::shared_ptr<PointCloud> source,
        std::shared_ptr<PointCloud> target,
        const glm::mat4& initial_transform,
        float correspondence_threshold,
        int min_correspondences = 3)
    {
        Reg3D::UmeyamaResult result;

        std::cout << "\n============================================" << std::endl;
        std::cout << "  Umeyama Refinement (Scale Estimation)" << std::endl;
        std::cout << "============================================" << std::endl;

        std::vector<glm::vec3> source_points, target_points;
        extractCorrespondences(source, target, initial_transform,
                               correspondence_threshold, source_points, target_points);

        result.correspondenceCount = static_cast<int>(source_points.size());
        std::cout << "  Correspondences found: " << result.correspondenceCount << std::endl;

        if (result.correspondenceCount < min_correspondences) {
            std::cerr << "   Not enough correspondences for Umeyama" << std::endl;
            result.transformation = initial_transform;
            result.scale = 1.0f;
            result.success = false;
            return result;
        }

        result.transformation = Reg3D::UmeyamaRegistration(source_points, target_points);
        result.scale = glm::length(glm::vec3(result.transformation[0]));

        float total_error = 0.0f;
        for (size_t i = 0; i < source_points.size(); i++) {
            glm::vec4 transformed = result.transformation * glm::vec4(source_points[i], 1.0f);
            total_error += glm::distance(glm::vec3(transformed), target_points[i]);
        }
        result.averageError = total_error / source_points.size();

        std::cout << "  Estimated scale: " << result.scale << std::endl;
        std::cout << "  Average error: " << result.averageError << " units" << std::endl;

        if (result.scale < 0.5f || result.scale > 2.0f) {
            std::cerr << "   Scale out of valid range (0.5 - 2.0)" << std::endl;
            result.transformation = initial_transform;
            result.scale = 1.0f;
            result.success = false;
            return result;
        }

        result.success = true;
        std::cout << "   Umeyama refinement successful" << std::endl;

        return result;
    }

    float estimateScale(
        std::shared_ptr<PointCloud> source,
        std::shared_ptr<PointCloud> target,
        const glm::mat4& transformation,
        float correspondence_threshold,
        int min_correspondences = 10)
    {
        std::vector<glm::vec3> source_points, target_points;
        extractCorrespondences(source, target, transformation,
                               correspondence_threshold, source_points, target_points);

        std::cout << "  Found " << source_points.size() << " correspondences" << std::endl;

        if (static_cast<int>(source_points.size()) < min_correspondences) {
            std::cout << "   Not enough correspondences for scale estimation" << std::endl;
            return 1.0f;
        }

        return Reg3D::estimateScaleOnly(source_points, target_points);
    }
};

inline std::shared_ptr<PointCloud> extractTargetPointCloud(
    const mCutMesh& screenMeshRef,
    int gridW, int gridH, float zThreshold)
{
    NoOpen3DRegistration tempReg;
    auto targetCloud = tempReg.extractFrontFacePoints(
        screenMeshRef, gridW, gridH, zThreshold);

    std::cout << "  [Custom] Target points: " << targetCloud->size() << std::endl;

    if (targetCloud->size() < 50) {
        std::cerr << " [Custom] ERROR: Not enough target points ("
                  << targetCloud->size() << " < 50)" << std::endl;
        return nullptr;
    }

    return targetCloud;
}

struct SourceBuildResult {
    std::shared_ptr<PointCloud> cloud;
    std::vector<std::vector<size_t>> surfaceVertexIndices;
    int totalSurfaceCount  = 0;
    int totalInternalCount = 0;
    double buildTimeMs     = 0.0;
};

inline SourceBuildResult buildSourcePointCloud(
    std::vector<mCutMesh*>& organMeshes,
    std::vector<std::string>& meshNames,
    bool forceRebuild = false)
{
    using namespace Reg3D;

    SourceBuildResult result;
    result.cloud = std::make_shared<PointCloud>();
    auto build_start = std::chrono::high_resolution_clock::now();

    bool use_cache = isCacheValid(meshNames) && !forceRebuild;

    if (!use_cache) {
        std::cout << "  [Custom] Building BVH trees and computing surface vertices..." << std::endl;

        g_surfaceCache.surfaceVertexIndices.clear();
        g_surfaceCache.surfaceVertexIndices.resize(organMeshes.size());
        g_surfaceCache.bvhTrees.clear();
        g_surfaceCache.bvhTrees.resize(organMeshes.size());
        g_surfaceCache.cachedMeshNames = meshNames;
        g_surfaceCache.meshCount = organMeshes.size();

        auto bvh_start = std::chrono::high_resolution_clock::now();
        for (size_t m = 0; m < organMeshes.size(); m++) {
            g_surfaceCache.bvhTrees[m].build(
                organMeshes[m]->mVertices, organMeshes[m]->mIndices);
            std::cout << "    [" << (m + 1) << "/" << organMeshes.size()
                      << "] BVH built for " << meshNames[m]
                      << " (" << g_surfaceCache.bvhTrees[m].nodes.size()
                      << " nodes)" << std::endl;
        }
        auto bvh_end = std::chrono::high_resolution_clock::now();
        auto bvh_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            bvh_end - bvh_start);
        std::cout << "  [Custom] BVH construction: " << bvh_ms.count() << "ms" << std::endl;

        std::cout << "  [Custom] Extracting surface vertices..." << std::endl;

        for (size_t m = 0; m < organMeshes.size(); m++) {
            std::cout << "  [" << (m + 1) << "/" << organMeshes.size()
            << "] Processing " << meshNames[m] << "..." << std::endl;

            int surface_count  = 0;
            int internal_count = 0;
            auto test_start = std::chrono::high_resolution_clock::now();

            size_t vertCount = organMeshes[m]->mVertices.size() / 3;
            for (size_t i = 0; i < vertCount; i++) {
                glm::vec3 vertex(
                    organMeshes[m]->mVertices[i * 3],
                    organMeshes[m]->mVertices[i * 3 + 1],
                    organMeshes[m]->mVertices[i * 3 + 2]);

                bool is_internal = false;
                for (size_t other_m = 0; other_m < organMeshes.size(); other_m++) {
                    if (other_m == m) continue;
                    if (isPointInsideMeshUsingBVH(vertex, g_surfaceCache.bvhTrees[other_m])) {
                        is_internal = true;
                        break;
                    }
                }

                if (!is_internal) {
                    g_surfaceCache.surfaceVertexIndices[m].push_back(i);
                    surface_count++;
                } else {
                    internal_count++;
                }
            }

            auto test_end = std::chrono::high_resolution_clock::now();
            auto test_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                test_end - test_start);

            std::cout << "    Surface: " << surface_count
                      << ", Internal: " << internal_count
                      << " (" << test_ms.count() << "ms)" << std::endl;

            result.totalSurfaceCount  += surface_count;
            result.totalInternalCount += internal_count;
        }

        g_surfaceCache.isValid = true;
        std::cout << "  [Custom] Cache updated" << std::endl;

    } else {
        std::cout << "  [Custom] Using cached BVH and surface indices" << std::endl;
    }

    result.surfaceVertexIndices = g_surfaceCache.surfaceVertexIndices;

    for (size_t m = 0; m < organMeshes.size(); m++) {
        for (size_t idx : g_surfaceCache.surfaceVertexIndices[m]) {
            glm::vec3 vertex(
                organMeshes[m]->mVertices[idx * 3],
                organMeshes[m]->mVertices[idx * 3 + 1],
                organMeshes[m]->mVertices[idx * 3 + 2]);
            result.cloud->points.push_back(vertex);

            if (!organMeshes[m]->mNormals.empty() &&
                organMeshes[m]->mNormals.size() > idx * 3 + 2) {
                result.cloud->normals.push_back(glm::vec3(
                    organMeshes[m]->mNormals[idx * 3],
                    organMeshes[m]->mNormals[idx * 3 + 1],
                    organMeshes[m]->mNormals[idx * 3 + 2]));
            }
        }
    }

    auto build_end = std::chrono::high_resolution_clock::now();
    result.buildTimeMs = std::chrono::duration_cast<std::chrono::microseconds>(
                             build_end - build_start).count() / 1000.0;

    std::cout << "  [Custom] Total surface points: " << result.cloud->size()
              << " (" << result.buildTimeMs << "ms)" << std::endl;

    return result;
}

struct ComparisonResult {
    bool   match       = false;
    size_t countA      = 0;
    size_t countB      = 0;
    int    mismatches  = 0;
    float  maxDiff     = 0.0f;
    float  avgDiff     = 0.0f;
};

inline ComparisonResult comparePointVectors(
    const std::vector<glm::vec3>& a,
    const std::vector<glm::vec3>& b,
    float tolerance = 1e-5f)
{
    ComparisonResult r;
    r.countA = a.size();
    r.countB = b.size();

    if (a.size() != b.size()) {
        r.match = false;
        std::cout << "   Size mismatch: " << r.countA << " vs " << r.countB << std::endl;
        return r;
    }

    float totalDiff = 0.0f;
    r.mismatches = 0;
    r.maxDiff = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        float diff = glm::length(a[i] - b[i]);
        totalDiff += diff;
        if (diff > r.maxDiff) r.maxDiff = diff;
        if (diff > tolerance) r.mismatches++;
    }

    r.avgDiff = (a.empty()) ? 0.0f : totalDiff / a.size();
    r.match = (r.mismatches == 0);

    return r;
}

inline void printComparison(
    const std::string& stageName,
    const ComparisonResult& r)
{
    std::cout << "\n  ========== " << stageName << " Comparison ==========" << std::endl;
    std::cout << "  Points:     Open3D=" << r.countA
              << "  Custom=" << r.countB << std::endl;

    if (r.countA != r.countB) {
        std::cout << "   MISMATCH: point count differs!" << std::endl;
        return;
    }

    std::cout << "  Max diff:   " << r.maxDiff << std::endl;
    std::cout << "  Avg diff:   " << r.avgDiff << std::endl;
    std::cout << "  Mismatches: " << r.mismatches << " / " << r.countA << std::endl;

    if (r.match) {
        std::cout << "   MATCH: Results are identical" << std::endl;
    } else {
        std::cout << "   MISMATCH: " << r.mismatches << " points differ" << std::endl;
    }
    std::cout << "  ==========================================" << std::endl;
}

inline ComparisonResult comparePointClouds(
    const PointCloud& custom,
    const std::vector<glm::vec3>& open3dPoints,
    const std::string& label = "PointCloud")
{
    auto r = comparePointVectors(open3dPoints, custom.points);
    printComparison(label, r);
    return r;
}

inline void debugStage1(
    const mCutMesh& screenMesh,
    int gridW, int gridH, float zThreshold,
    const std::vector<glm::vec3>& open3dTargetPoints)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 1: Target Point Cloud Extraction       |" << std::endl;
    std::cout << "================================================" << std::endl;

    auto customCloud = extractTargetPointCloud(screenMesh, gridW, gridH, zThreshold);
    if (!customCloud) {
        std::cerr << "  [Custom] Failed to extract target points" << std::endl;
        return;
    }

    if (!open3dTargetPoints.empty()) {
        auto result = comparePointVectors(open3dTargetPoints, customCloud->points);
        printComparison("Stage 1: Target Points", result);
    } else {
        std::cout << "  (Open3D results not provided, skipping comparison)" << std::endl;
    }

    std::cout << "\n  Sample points (first 5):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, customCloud->size()); i++) {
        const auto& p = customCloud->points[i];
        std::cout << "    [" << i << "] (" << p.x << ", " << p.y << ", " << p.z << ")";

        if (i < open3dTargetPoints.size()) {
            const auto& o = open3dTargetPoints[i];
            std::cout << "  Open3D: (" << o.x << ", " << o.y << ", " << o.z << ")";
        }
        std::cout << std::endl;
    }
}

inline void debugStage2(
    std::vector<mCutMesh*>& organMeshes,
    std::vector<std::string>& meshNames,
    const std::vector<glm::vec3>& open3dSourcePoints)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 2: Source Point Cloud (BVH)            |" << std::endl;
    std::cout << "================================================" << std::endl;

    auto buildResult = buildSourcePointCloud(organMeshes, meshNames);

    std::cout << "\n  Build summary:" << std::endl;
    std::cout << "    Total surface:  " << buildResult.totalSurfaceCount << std::endl;
    std::cout << "    Total internal: " << buildResult.totalInternalCount << std::endl;
    std::cout << "    Cloud size:     " << buildResult.cloud->size() << std::endl;
    std::cout << "    Build time:     " << buildResult.buildTimeMs << "ms" << std::endl;

    std::cout << "\n  Per-mesh breakdown:" << std::endl;
    for (size_t m = 0; m < meshNames.size(); m++) {
        std::cout << "    " << meshNames[m] << ": "
                  << buildResult.surfaceVertexIndices[m].size()
                  << " surface vertices" << std::endl;
    }

    if (!open3dSourcePoints.empty()) {
        auto result = comparePointVectors(open3dSourcePoints, buildResult.cloud->points);
        printComparison("Stage 2: Source Points", result);
    } else {
        std::cout << "\n  (Open3D results not provided, skipping comparison)" << std::endl;
    }

    std::cout << "\n  Sample points (first 5):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, buildResult.cloud->size()); i++) {
        const auto& p = buildResult.cloud->points[i];
        std::cout << "    [" << i << "] (" << p.x << ", " << p.y << ", " << p.z << ")";

        if (i < open3dSourcePoints.size()) {
            const auto& o = open3dSourcePoints[i];
            std::cout << "  Open3D: (" << o.x << ", " << o.y << ", " << o.z << ")";
        }
        std::cout << std::endl;
    }
}

inline void debugStage3(
    const mCutMesh& screenMesh,
    int gridW, int gridH, float zThreshold,
    float voxel_size,
    const std::vector<glm::vec3>& open3dDownPoints)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 3: Voxel Down Sampling                 |" << std::endl;
    std::cout << "================================================" << std::endl;

    NoOpen3DRegistration reg;
    auto cloud = reg.extractFrontFacePoints(screenMesh, gridW, gridH, zThreshold);
    auto downsampled = reg.voxelDownSample(cloud, voxel_size);

    std::cout << "  Open3D downsampled:  " << open3dDownPoints.size() << " points" << std::endl;
    std::cout << "  Custom downsampled:  " << downsampled->size() << " points" << std::endl;

    if (open3dDownPoints.empty()) {
        std::cout << "  (Open3D results not provided, skipping comparison)" << std::endl;
        return;
    }

    int count_diff = static_cast<int>(downsampled->size()) - static_cast<int>(open3dDownPoints.size());
    std::cout << "  Count difference:    " << count_diff
              << " (" << std::abs(count_diff) * 100.0f / open3dDownPoints.size() << "%)" << std::endl;

    if (!open3dDownPoints.empty() && !downsampled->empty()) {
        NanoflannAdaptor adaptor(open3dDownPoints);
        auto tree = buildKDTree(adaptor);

        float max_dist  = 0.0f;
        double total_dist = 0.0;
        int far_count = 0;

        for (size_t i = 0; i < downsampled->size(); i++) {
            size_t nn_idx;
            float nn_dist_sq;
            searchKNN1(*tree, downsampled->points[i], nn_idx, nn_dist_sq);

            float dist = std::sqrt(nn_dist_sq);
            total_dist += dist;
            if (dist > max_dist) max_dist = dist;
            if (dist > voxel_size * 0.5f) far_count++;
        }

        float avg_dist = static_cast<float>(total_dist / downsampled->size());

        std::cout << "\n  Set-based comparison (CustomOpen3D nearest):" << std::endl;
        std::cout << "    Max distance:   " << max_dist << std::endl;
        std::cout << "    Avg distance:   " << avg_dist << std::endl;
        std::cout << "    Far points:     " << far_count
                  << " / " << downsampled->size()
                  << " (> " << voxel_size * 0.5f << ")" << std::endl;

        if (max_dist < voxel_size * 0.6f && far_count == 0) {
            std::cout << "     GOOD: All points within expected voxel range" << std::endl;
        } else if (far_count < static_cast<int>(downsampled->size()) * 0.05) {
            std::cout << "     ACCEPTABLE: <5% outlier points" << std::endl;
        } else {
            std::cout << "     MISMATCH: Significant differences detected" << std::endl;
        }
    }

    std::cout << "\n  Custom sample (first 5):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, downsampled->size()); i++) {
        const auto& p = downsampled->points[i];
        std::cout << "    [" << i << "] (" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
    }
    std::cout << "  Open3D sample (first 5):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, open3dDownPoints.size()); i++) {
        const auto& p = open3dDownPoints[i];
        std::cout << "    [" << i << "] (" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
    }
}

inline void debugStage4(
    const mCutMesh& screenMesh,
    int gridW, int gridH, float zThreshold,
    float voxel_size,
    const std::vector<glm::vec3>& open3dDownPoints,
    const std::vector<glm::vec3>& open3dNormals)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 4: Normal Estimation                   |" << std::endl;
    std::cout << "================================================" << std::endl;

    NoOpen3DRegistration reg;
    auto cloud = reg.extractFrontFacePoints(screenMesh, gridW, gridH, zThreshold);
    auto downsampled = reg.voxelDownSample(cloud, voxel_size);

    reg.estimateNormals(downsampled, voxel_size * 2.0f, 30);

    std::cout << "  Custom points:  " << downsampled->size()
              << ", normals: " << downsampled->normals.size() << std::endl;
    std::cout << "  Open3D points:  " << open3dDownPoints.size()
              << ", normals: " << open3dNormals.size() << std::endl;

    if (open3dDownPoints.empty() || open3dNormals.empty()) {
        std::cout << "  (Open3D results not provided, skipping comparison)" << std::endl;
        return;
    }

    NanoflannAdaptor adaptor(open3dDownPoints);
    auto tree = buildKDTree(adaptor);

    float total_cos = 0.0f;
    float min_cos = 1.0f;
    int aligned_count = 0;
    int misaligned_count = 0;
    int matched = 0;

    for (size_t i = 0; i < downsampled->size(); i++) {
        size_t nn_idx;
        float nn_dist_sq;
        searchKNN1(*tree, downsampled->points[i], nn_idx, nn_dist_sq);

        if (nn_dist_sq < voxel_size * voxel_size * 0.01f) {
            float dot = std::abs(glm::dot(downsampled->normals[i], open3dNormals[nn_idx]));
            dot = std::min(dot, 1.0f);

            total_cos += dot;
            if (dot < min_cos) min_cos = dot;
            if (dot > 0.9f) aligned_count++;
            if (dot < 0.5f) misaligned_count++;
            matched++;
        }
    }

    if (matched > 0) {
        float avg_cos = total_cos / matched;
        float avg_angle_deg = std::acos(std::min(avg_cos, 1.0f)) * 180.0f / static_cast<float>(M_PI);
        float worst_angle_deg = std::acos(std::min(min_cos, 1.0f)) * 180.0f / static_cast<float>(M_PI);

        std::cout << "\n  Normal comparison (" << matched << " matched points):" << std::endl;
        std::cout << "    Avg |cos|:      " << avg_cos
                  << " (angle: " << avg_angle_deg << "deg)" << std::endl;
        std::cout << "    Worst |cos|:    " << min_cos
                  << " (angle: " << worst_angle_deg << "deg)" << std::endl;
        std::cout << "    Aligned (>0.9): " << aligned_count
                  << " / " << matched
                  << " (" << (aligned_count * 100.0f / matched) << "%)" << std::endl;
        std::cout << "    Misaligned(<0.5):" << misaligned_count
                  << " / " << matched << std::endl;

        if (avg_cos > 0.95f && misaligned_count == 0) {
            std::cout << "     GOOD: Normals closely match Open3D" << std::endl;
        } else if (avg_cos > 0.8f) {
            std::cout << "     ACCEPTABLE: Most normals aligned" << std::endl;
        } else {
            std::cout << "     MISMATCH: Significant normal differences" << std::endl;
        }
    }

    std::cout << "\n  Sample normals (first 5):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, downsampled->size()); i++) {
        const auto& n = downsampled->normals[i];
        std::cout << "    Custom [" << i << "] (" << n.x << ", " << n.y << ", " << n.z << ")";

        size_t nn_idx;
        float nn_dist_sq;
        searchKNN1(*tree, downsampled->points[i], nn_idx, nn_dist_sq);
        if (nn_dist_sq < voxel_size * voxel_size * 0.01f && nn_idx < open3dNormals.size()) {
            const auto& on = open3dNormals[nn_idx];
            float dot = glm::dot(n, on);
            std::cout << "  Open3D: (" << on.x << ", " << on.y << ", " << on.z
                      << ")  dot=" << dot;
        }
        std::cout << std::endl;
    }
}

inline void debugStage5(
    const mCutMesh& screenMesh,
    int gridW, int gridH, float zThreshold,
    float voxel_size,
    const std::vector<glm::vec3>& open3dDownPoints,
    const std::vector<std::vector<float>>& open3dFPFH)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 5: FPFH Feature Computation            |" << std::endl;
    std::cout << "================================================" << std::endl;

    NoOpen3DRegistration reg;
    auto cloud = reg.extractFrontFacePoints(screenMesh, gridW, gridH, zThreshold);
    auto downsampled = reg.voxelDownSample(cloud, voxel_size);
    reg.estimateNormals(downsampled, voxel_size * 2.0f, 30);

    auto fpfh = reg.computeFPFH(downsampled, voxel_size);

    std::cout << "  Custom:  " << fpfh->numPoints() << " points, dim=" << fpfh->dimension << std::endl;
    std::cout << "  Open3D:  " << open3dFPFH.size() << " points" << std::endl;

    if (open3dDownPoints.empty() || open3dFPFH.empty()) {
        std::cout << "  (Open3D results not provided, skipping comparison)" << std::endl;

        float total_norm = 0.0f;
        float max_norm = 0.0f;
        for (size_t i = 0; i < fpfh->numPoints(); i++) {
            float norm = 0.0f;
            for (int d = 0; d < 33; d++) {
                norm += fpfh->data[i][d] * fpfh->data[i][d];
            }
            norm = std::sqrt(norm);
            total_norm += norm;
            if (norm > max_norm) max_norm = norm;
        }
        float avg_norm = fpfh->numPoints() > 0 ? total_norm / fpfh->numPoints() : 0.0f;
        std::cout << "  Custom feature stats: avg_norm=" << avg_norm
                  << ", max_norm=" << max_norm << std::endl;
        return;
    }

    NanoflannAdaptor adaptor(open3dDownPoints);
    auto tree = buildKDTree(adaptor);

    float total_l2 = 0.0f;
    float max_l2 = 0.0f;
    float total_cosine = 0.0f;
    float min_cosine = 1.0f;
    int matched = 0;
    int good_count = 0;

    std::vector<float> dim_total_diff(33, 0.0f);
    std::vector<float> dim_max_diff(33, 0.0f);

    for (size_t i = 0; i < downsampled->size(); i++) {
        size_t nn_idx;
        float nn_dist_sq;
        searchKNN1(*tree, downsampled->points[i], nn_idx, nn_dist_sq);

        if (nn_dist_sq > voxel_size * voxel_size * 0.01f) continue;
        if (nn_idx >= open3dFPFH.size()) continue;

        matched++;

        const auto& custom_feat = fpfh->data[i];
        const auto& o3d_feat = open3dFPFH[nn_idx];

        float l2 = 0.0f;
        for (int d = 0; d < 33; d++) {
            float diff = custom_feat[d] - o3d_feat[d];
            l2 += diff * diff;
            float abs_diff = std::abs(diff);
            dim_total_diff[d] += abs_diff;
            if (abs_diff > dim_max_diff[d]) dim_max_diff[d] = abs_diff;
        }
        l2 = std::sqrt(l2);
        total_l2 += l2;
        if (l2 > max_l2) max_l2 = l2;

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int d = 0; d < 33; d++) {
            dot    += custom_feat[d] * o3d_feat[d];
            norm_a += custom_feat[d] * custom_feat[d];
            norm_b += o3d_feat[d] * o3d_feat[d];
        }
        float cosine = 0.0f;
        if (norm_a > 1e-10f && norm_b > 1e-10f) {
            cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }
        total_cosine += cosine;
        if (cosine < min_cosine) min_cosine = cosine;
        if (cosine > 0.9f) good_count++;
    }

    if (matched > 0) {
        float avg_l2 = total_l2 / matched;
        float avg_cosine = total_cosine / matched;

        std::cout << "\n  FPFH comparison (" << matched << " matched points):" << std::endl;
        std::cout << "    Avg L2 distance:   " << avg_l2 << std::endl;
        std::cout << "    Max L2 distance:   " << max_l2 << std::endl;
        std::cout << "    Avg cosine sim:    " << avg_cosine << std::endl;
        std::cout << "    Min cosine sim:    " << min_cosine << std::endl;
        std::cout << "    Good (cos>0.9):    " << good_count << " / " << matched
                  << " (" << (good_count * 100.0f / matched) << "%)" << std::endl;

        std::cout << "\n    Per-dimension max diff (theta/alpha/phi):" << std::endl;
        float theta_max = 0, alpha_max = 0, phi_max = 0;
        for (int d = 0; d < 11; d++) theta_max = std::max(theta_max, dim_max_diff[d]);
        for (int d = 11; d < 22; d++) alpha_max = std::max(alpha_max, dim_max_diff[d]);
        for (int d = 22; d < 33; d++) phi_max = std::max(phi_max, dim_max_diff[d]);
        std::cout << "      theta bins:  max_diff=" << theta_max << std::endl;
        std::cout << "      alpha bins:  max_diff=" << alpha_max << std::endl;
        std::cout << "      phi bins:    max_diff=" << phi_max << std::endl;

        if (avg_cosine > 0.95f && good_count > matched * 0.9) {
            std::cout << "     GOOD: FPFH features closely match Open3D" << std::endl;
        } else if (avg_cosine > 0.8f) {
            std::cout << "     ACCEPTABLE: Most features similar" << std::endl;
        } else {
            std::cout << "     MISMATCH: Significant feature differences" << std::endl;
        }
    } else {
        std::cout << "  No matched points for comparison" << std::endl;
    }

    std::cout << "\n  Sample features (first 3 points, first 11 dims = theta):" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, downsampled->size()); i++) {
        std::cout << "    Custom [" << i << "]: ";
        for (int d = 0; d < 11; d++) {
            std::cout << std::fixed << std::setprecision(2) << fpfh->data[i][d];
            if (d < 10) std::cout << ", ";
        }
        std::cout << std::endl;

        size_t nn_idx;
        float nn_dist_sq;
        searchKNN1(*tree, downsampled->points[i], nn_idx, nn_dist_sq);
        if (nn_dist_sq < voxel_size * voxel_size * 0.01f && nn_idx < open3dFPFH.size()) {
            std::cout << "    Open3D[" << nn_idx << "]: ";
            for (int d = 0; d < 11; d++) {
                std::cout << std::fixed << std::setprecision(2) << open3dFPFH[nn_idx][d];
                if (d < 10) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
}

inline void debugStage6(
    const mCutMesh& screenMesh,
    int gridW, int gridH, float zThreshold,
    float voxel_size,
    std::vector<mCutMesh*>& organMeshes,
    std::vector<std::string>& meshNames,
    float open3dFgrFitness = -1.0f,
    float open3dFgrRMSE = -1.0f,
    const glm::mat4& open3dFgrTransform = glm::mat4(1.0f),
    const std::vector<glm::vec3>& o3dSourcePts = {},
    const std::vector<glm::vec3>& o3dSourceNormals = {},
    const std::vector<std::vector<float>>& o3dSourceFPFH = {},
    const std::vector<glm::vec3>& o3dTargetPts = {},
    const std::vector<std::vector<float>>& o3dTargetFPFH = {})
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 6: Fast Global Registration (FGR)      |" << std::endl;
    std::cout << "================================================" << std::endl;

    NoOpen3DRegistration reg;

    auto targetCloud = reg.extractFrontFacePoints(screenMesh, gridW, gridH, zThreshold);
    auto targetDown = reg.preprocess(targetCloud, voxel_size, false);

    auto sourceResult = buildSourcePointCloud(organMeshes, meshNames);
    auto sourceDown = reg.preprocess(sourceResult.cloud, voxel_size, true);

    std::cout << "\n  Source: " << sourceDown->size() << " points" << std::endl;
    std::cout << "  Target: " << targetDown->size() << " points" << std::endl;

    if (!o3dSourcePts.empty()) {
        std::cout << "\n  --- Source Point Cloud Comparison ---" << std::endl;
        std::cout << "  Custom source: " << sourceDown->size()
                  << ", Open3D source: " << o3dSourcePts.size() << std::endl;

        if (sourceDown->size() == o3dSourcePts.size()) {
            float max_pt_diff = 0.0f;
            float avg_pt_diff = 0.0f;
            int matched = 0;
            for (size_t i = 0; i < sourceDown->size(); i++) {
                float best_dist = 1e10f;
                for (size_t j = 0; j < o3dSourcePts.size(); j++) {
                    float d = glm::length(sourceDown->points[i] - o3dSourcePts[j]);
                    if (d < best_dist) best_dist = d;
                }
                if (best_dist < 0.01f) matched++;
                avg_pt_diff += best_dist;
                if (best_dist > max_pt_diff) max_pt_diff = best_dist;
            }
            avg_pt_diff /= sourceDown->size();
            std::cout << "  Source pts matched (dist<0.01): " << matched
                      << "/" << sourceDown->size() << std::endl;
            std::cout << "  Avg nearest dist: " << avg_pt_diff
                      << ", Max: " << max_pt_diff << std::endl;
        }
    }

    if (!o3dSourceNormals.empty() && !o3dSourcePts.empty() &&
        sourceDown->size() == o3dSourcePts.size()) {
        std::cout << "\n  --- Source Normal Comparison ---" << std::endl;

        std::vector<int> pt_map(sourceDown->size(), -1);
        for (size_t i = 0; i < sourceDown->size(); i++) {
            float best_d = 1e10f;
            int best_j = 0;
            for (size_t j = 0; j < o3dSourcePts.size(); j++) {
                float d = glm::length(sourceDown->points[i] - o3dSourcePts[j]);
                if (d < best_d) { best_d = d; best_j = static_cast<int>(j); }
            }
            if (best_d < 0.01f) pt_map[i] = best_j;
        }

        int matched = 0, aligned = 0, flipped = 0, misaligned = 0;
        float sum_abs_cos = 0, worst_abs_cos = 1.0f;
        for (size_t i = 0; i < sourceDown->size(); i++) {
            if (pt_map[i] < 0 || pt_map[i] >= static_cast<int>(o3dSourceNormals.size())) continue;
            matched++;

            glm::vec3 nc = glm::normalize(sourceDown->normals[i]);
            glm::vec3 no = glm::normalize(o3dSourceNormals[pt_map[i]]);
            float dot = glm::dot(nc, no);
            float abs_dot = std::abs(dot);

            sum_abs_cos += abs_dot;
            if (abs_dot < worst_abs_cos) worst_abs_cos = abs_dot;

            if (dot > 0.9f) aligned++;
            else if (dot < -0.9f) flipped++;
            else misaligned++;
        }

        float avg_abs_cos = (matched > 0) ? sum_abs_cos / matched : 0;
        std::cout << "  Matched: " << matched << "/" << sourceDown->size() << std::endl;
        std::cout << "  Avg |cos|: " << avg_abs_cos << std::endl;
        std::cout << "  Worst |cos|: " << worst_abs_cos << std::endl;
        std::cout << "  Aligned (dot>0.9): " << aligned << std::endl;
        std::cout << "  Flipped (dot<-0.9): " << flipped << std::endl;
        std::cout << "  Misaligned: " << misaligned << std::endl;

        std::cout << "  Sample normals (first 5 matched):" << std::endl;
        int shown = 0;
        for (size_t i = 0; i < sourceDown->size() && shown < 5; i++) {
            if (pt_map[i] < 0) continue;
            glm::vec3 nc = sourceDown->normals[i];
            glm::vec3 no = o3dSourceNormals[pt_map[i]];
            float dot = glm::dot(glm::normalize(nc), glm::normalize(no));
            std::cout << "    [" << i << "] Custom(" << nc.x << "," << nc.y << "," << nc.z
                      << ")  O3D(" << no.x << "," << no.y << "," << no.z
                      << ")  dot=" << dot << std::endl;
            shown++;
        }

        if (avg_abs_cos > 0.95f && misaligned == 0) {
            if (flipped > aligned) {
                std::cout << "   Normals mostly FLIPPED but consistent direction" << std::endl;
            } else {
                std::cout << "   Source normals match Open3D" << std::endl;
            }
        } else {
            std::cout << "   Source normals DIFFER from Open3D!" << std::endl;
            if (misaligned > matched / 2) {
                std::cout << "     MST orientation propagation taking different path" << std::endl;
            }
        }
    }

    auto sourceFpfh = reg.computeFPFH(sourceDown, voxel_size);
    auto targetFpfh = reg.computeFPFH(targetDown, voxel_size);

    if (!o3dSourceFPFH.empty() && !o3dSourcePts.empty() &&
        sourceDown->size() == o3dSourcePts.size()) {
        std::cout << "\n  --- Source FPFH Comparison ---" << std::endl;

        std::vector<int> pt_mapping(sourceDown->size(), -1);
        for (size_t i = 0; i < sourceDown->size(); i++) {
            float best_dist = 1e10f;
            int best_j = -1;
            for (size_t j = 0; j < o3dSourcePts.size(); j++) {
                float d = glm::length(sourceDown->points[i] - o3dSourcePts[j]);
                if (d < best_dist) {
                    best_dist = d;
                    best_j = static_cast<int>(j);
                }
            }
            if (best_dist < 0.01f) pt_mapping[i] = best_j;
        }

        int mapped = 0;
        float avg_l2 = 0, max_l2 = 0;
        float avg_cos = 0, min_cos = 1.0f;
        for (size_t i = 0; i < sourceDown->size(); i++) {
            if (pt_mapping[i] < 0) continue;
            int j = pt_mapping[i];
            if (j >= static_cast<int>(o3dSourceFPFH.size())) continue;
            mapped++;

            float l2 = 0, dot = 0, norm_a = 0, norm_b = 0;
            for (int d = 0; d < 33; d++) {
                float a = sourceFpfh->data[i][d];
                float b = o3dSourceFPFH[j][d];
                l2 += (a - b) * (a - b);
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            l2 = std::sqrt(l2);
            float cos_sim = (norm_a > 0 && norm_b > 0)
                                ? dot / (std::sqrt(norm_a) * std::sqrt(norm_b)) : 0.0f;

            avg_l2 += l2;
            if (l2 > max_l2) max_l2 = l2;
            avg_cos += cos_sim;
            if (cos_sim < min_cos) min_cos = cos_sim;
        }
        if (mapped > 0) {
            avg_l2 /= mapped;
            avg_cos /= mapped;
        }

        std::cout << "  Matched source pts: " << mapped << "/" << sourceDown->size() << std::endl;
        std::cout << "  Source FPFH L2:  avg=" << avg_l2 << ", max=" << max_l2 << std::endl;
        std::cout << "  Source FPFH cos: avg=" << avg_cos << ", min=" << min_cos << std::endl;

        if (avg_cos > 0.99f) {
            std::cout << "   Source FPFH matches Open3D" << std::endl;
        } else {
            std::cout << "   Source FPFH DIFFERS from Open3D!" << std::endl;
        }
    }

    if (!o3dTargetFPFH.empty() && !o3dTargetPts.empty() &&
        targetDown->size() == o3dTargetPts.size()) {
        std::cout << "\n  --- Target FPFH Comparison ---" << std::endl;

        std::vector<int> pt_mapping(targetDown->size(), -1);
        for (size_t i = 0; i < targetDown->size(); i++) {
            float best_dist = 1e10f;
            int best_j = -1;
            for (size_t j = 0; j < o3dTargetPts.size(); j++) {
                float d = glm::length(targetDown->points[i] - o3dTargetPts[j]);
                if (d < best_dist) {
                    best_dist = d;
                    best_j = static_cast<int>(j);
                }
            }
            if (best_dist < 0.01f) pt_mapping[i] = best_j;
        }

        int mapped = 0;
        float avg_l2 = 0, max_l2 = 0;
        float avg_cos = 0, min_cos = 1.0f;
        for (size_t i = 0; i < targetDown->size(); i++) {
            if (pt_mapping[i] < 0) continue;
            int j = pt_mapping[i];
            if (j >= static_cast<int>(o3dTargetFPFH.size())) continue;
            mapped++;

            float l2 = 0, dot = 0, norm_a = 0, norm_b = 0;
            for (int d = 0; d < 33; d++) {
                float a = targetFpfh->data[i][d];
                float b = o3dTargetFPFH[j][d];
                l2 += (a - b) * (a - b);
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            l2 = std::sqrt(l2);
            float cos_sim = (norm_a > 0 && norm_b > 0)
                                ? dot / (std::sqrt(norm_a) * std::sqrt(norm_b)) : 0.0f;

            avg_l2 += l2;
            if (l2 > max_l2) max_l2 = l2;
            avg_cos += cos_sim;
            if (cos_sim < min_cos) min_cos = cos_sim;
        }
        if (mapped > 0) {
            avg_l2 /= mapped;
            avg_cos /= mapped;
        }

        std::cout << "  Matched target pts: " << mapped << "/" << targetDown->size() << std::endl;
        std::cout << "  Target FPFH L2:  avg=" << avg_l2 << ", max=" << max_l2 << std::endl;
        std::cout << "  Target FPFH cos: avg=" << avg_cos << ", min=" << min_cos << std::endl;

        if (avg_cos > 0.99f) {
            std::cout << "   Target FPFH matches Open3D" << std::endl;
        } else {
            std::cout << "   Target FPFH DIFFERS from Open3D!" << std::endl;
        }
    }

    auto fgrResult = reg.fastGlobalRegistration(
        sourceDown, targetDown, sourceFpfh, targetFpfh, voxel_size);

    {
        bool is_identity = true;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                float expected = (r == c) ? 1.0f : 0.0f;
                if (std::abs(fgrResult.transformation[c][r] - expected) > 0.001f) {
                    is_identity = false;
                    break;
                }
            }
            if (!is_identity) break;
        }

        std::cout << "\n  === FGR Flow Summary ===" << std::endl;
        if (fgrResult.fitness < 0.001f && is_identity) {
            std::cout << "    Result: Identity (tuple test -> 0 corres -> < 10 -> no GNC)" << std::endl;
            std::cout << "    This matches Open3D behavior when tuple test fails." << std::endl;
        } else {
            std::cout << "    Result: GNC optimization produced a transform" << std::endl;
            std::cout << "    Fitness: " << fgrResult.fitness << ", RMSE: " << fgrResult.inlier_rmse << std::endl;
        }
        std::cout << "  =========================" << std::endl;
    }

    RegistrationResult fgrWithO3dFpfh;
    bool ran_with_o3d_fpfh = false;

    if (!o3dSourceFPFH.empty() && !o3dTargetFPFH.empty() &&
        !o3dSourcePts.empty() && !o3dTargetPts.empty()) {

        std::cout << "\n  === FGR with Open3D FPFH (diagnostic test) ===" << std::endl;

        auto srcFpfhO3d = std::make_shared<FeatureSet>();
        srcFpfhO3d->dimension = 33;
        srcFpfhO3d->data.resize(sourceDown->size());

        for (size_t i = 0; i < sourceDown->size(); i++) {
            float best_dist = 1e10f;
            int best_j = 0;
            for (size_t j = 0; j < o3dSourcePts.size(); j++) {
                float d = glm::length(sourceDown->points[i] - o3dSourcePts[j]);
                if (d < best_dist) { best_dist = d; best_j = static_cast<int>(j); }
            }
            if (best_j < static_cast<int>(o3dSourceFPFH.size())) {
                srcFpfhO3d->data[i] = o3dSourceFPFH[best_j];
            } else {
                srcFpfhO3d->data[i].resize(33, 0.0f);
            }
        }

        auto tgtFpfhO3d = std::make_shared<FeatureSet>();
        tgtFpfhO3d->dimension = 33;
        tgtFpfhO3d->data.resize(targetDown->size());

        for (size_t i = 0; i < targetDown->size(); i++) {
            float best_dist = 1e10f;
            int best_j = 0;
            for (size_t j = 0; j < o3dTargetPts.size(); j++) {
                float d = glm::length(targetDown->points[i] - o3dTargetPts[j]);
                if (d < best_dist) { best_dist = d; best_j = static_cast<int>(j); }
            }
            if (best_j < static_cast<int>(o3dTargetFPFH.size())) {
                tgtFpfhO3d->data[i] = o3dTargetFPFH[best_j];
            } else {
                tgtFpfhO3d->data[i].resize(33, 0.0f);
            }
        }

        fgrWithO3dFpfh = reg.fastGlobalRegistration(
            sourceDown, targetDown, srcFpfhO3d, tgtFpfhO3d, voxel_size);
        ran_with_o3d_fpfh = true;

        std::cout << "  [O3D FPFH -> Custom FGR] Fitness: " << fgrWithO3dFpfh.fitness
                  << ", RMSE: " << fgrWithO3dFpfh.inlier_rmse << std::endl;
    }

    std::cout << "\n  ========== FGR Comparison ==========" << std::endl;

    std::cout << "  Custom FGR:" << std::endl;
    std::cout << "    Fitness: " << fgrResult.fitness << std::endl;
    std::cout << "    RMSE:    " << fgrResult.inlier_rmse << std::endl;

    if (open3dFgrFitness >= 0.0f) {
        std::cout << "  Open3D FGR:" << std::endl;
        std::cout << "    Fitness: " << open3dFgrFitness << std::endl;
        std::cout << "    RMSE:    " << open3dFgrRMSE << std::endl;

        float fitness_diff = std::abs(fgrResult.fitness - open3dFgrFitness);
        float rmse_diff = std::abs(fgrResult.inlier_rmse - open3dFgrRMSE);

        std::cout << "\n  Differences:" << std::endl;
        std::cout << "    Fitness diff:  " << fitness_diff << std::endl;
        std::cout << "    RMSE diff:     " << rmse_diff << std::endl;

        float max_t_diff = 0.0f;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float diff = std::abs(fgrResult.transformation[c][r] - open3dFgrTransform[c][r]);
                if (diff > max_t_diff) max_t_diff = diff;
            }
        }
        std::cout << "    Max matrix element diff: " << max_t_diff << std::endl;

        std::cout << "\n    Custom Transform:                Open3D Transform:" << std::endl;
        for (int row = 0; row < 4; row++) {
            std::cout << "    [";
            for (int col = 0; col < 4; col++) {
                std::cout << std::fixed << std::setprecision(4)
                << std::setw(9) << fgrResult.transformation[col][row];
            }
            std::cout << " ]  [";
            for (int col = 0; col < 4; col++) {
                std::cout << std::fixed << std::setprecision(4)
                << std::setw(9) << open3dFgrTransform[col][row];
            }
            std::cout << " ]" << std::endl;
        }

        if (max_t_diff < 0.01f) {
            std::cout << "\n    ** EXACT MATCH: Transform matrices are identical!" << std::endl;
        } else if (fitness_diff < 0.01f && max_t_diff < 0.1f) {
            std::cout << "\n    ** GOOD: FGR results closely match Open3D" << std::endl;
        } else if (fitness_diff < 0.05f && rmse_diff < 0.1f) {
            std::cout << "\n    * ACCEPTABLE: FGR results reasonably close" << std::endl;
            std::cout << "    (FGR has randomness in tuple test, some variance is normal)" << std::endl;
        } else {
            std::cout << "\n    X MISMATCH: Significant differences" << std::endl;
            std::cout << "    (Note: Different random seeds cause different tuple selections)" << std::endl;
        }

        if (ran_with_o3d_fpfh) {
            std::cout << "\n  --- Diagnostic: Custom FGR + Open3D FPFH ---" << std::endl;
            std::cout << "    Fitness: " << fgrWithO3dFpfh.fitness
                      << "  (Open3D: " << open3dFgrFitness << ")" << std::endl;
            std::cout << "    RMSE:    " << fgrWithO3dFpfh.inlier_rmse
                      << "  (Open3D: " << open3dFgrRMSE << ")" << std::endl;

            float diag_fitness_diff = std::abs(fgrWithO3dFpfh.fitness - open3dFgrFitness);
            float diag_max_t = 0.0f;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) {
                    float d = std::abs(fgrWithO3dFpfh.transformation[c][r] - open3dFgrTransform[c][r]);
                    if (d > diag_max_t) diag_max_t = d;
                }
            std::cout << "    Fitness diff: " << diag_fitness_diff << std::endl;
            std::cout << "    Max matrix diff: " << diag_max_t << std::endl;

            std::cout << "    O3D-FPFH Transform:              Open3D Transform:" << std::endl;
            for (int row = 0; row < 4; row++) {
                std::cout << "    [";
                for (int col = 0; col < 4; col++)
                    std::cout << std::fixed << std::setprecision(4)
                              << std::setw(9) << fgrWithO3dFpfh.transformation[col][row];
                std::cout << " ]  [";
                for (int col = 0; col < 4; col++)
                    std::cout << std::fixed << std::setprecision(4)
                              << std::setw(9) << open3dFgrTransform[col][row];
                std::cout << " ]" << std::endl;
            }

            if (diag_max_t < 0.01f) {
                std::cout << "    ** EXACT MATCH with Open3D FPFH!" << std::endl;
            } else if (diag_fitness_diff < 0.05f && diag_max_t < 0.5f) {
                std::cout << "    * FGR algorithm is correct! Source FPFH is the problem." << std::endl;
            } else if (diag_fitness_diff < 0.05f) {
                std::cout << "    ~ FGR algorithm OK, tuple randomness causes transform diff" << std::endl;
            } else {
                std::cout << "    X FGR algorithm itself may have issues" << std::endl;
            }
        }
    } else {
        std::cout << "\n  (Open3D results not provided, skipping comparison)" << std::endl;
    }

    std::cout << "  =====================================" << std::endl;
}

inline void debugStage7(
    const mCutMesh& screenMesh,
    int gridW, int gridH, float zThreshold,
    float voxel_size,
    std::vector<mCutMesh*>& organMeshes,
    std::vector<std::string>& meshNames,
    const glm::mat4& customFgrTransform,
    const glm::mat4& open3dFgrTransform,
    float open3dIcpFitness = -1.0f,
    float open3dIcpRMSE = -1.0f,
    const glm::mat4& open3dIcpTransform = glm::mat4(1.0f),
    std::shared_ptr<PointCloud> o3dSourceDown = nullptr,
    std::shared_ptr<PointCloud> o3dTargetDown = nullptr)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 7: ICP Refinement                      |" << std::endl;
    std::cout << "================================================" << std::endl;

    NoOpen3DRegistration reg;

    auto targetCloud = reg.extractFrontFacePoints(screenMesh, gridW, gridH, zThreshold);
    auto targetDown = reg.preprocess(targetCloud, voxel_size, false);

    auto sourceResult = buildSourcePointCloud(organMeshes, meshNames);
    auto sourceDown = reg.preprocess(sourceResult.cloud, voxel_size, true);

    float icp_distance = voxel_size * 0.4f;

    std::cout << "  Source: " << sourceDown->size() << " points" << std::endl;
    std::cout << "  Target: " << targetDown->size() << " points" << std::endl;
    std::cout << "  ICP distance threshold: " << icp_distance << std::endl;
    std::cout << "  Target has normals: " << (targetDown->hasNormals() ? "YES" : "NO") << std::endl;

    std::cout << "\n  --- Test A: Custom ICP + Open3D FGR init ---" << std::endl;
    {
        std::cout << "  Init transform (Open3D FGR):" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", open3dFgrTransform[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        auto icpResult = reg.icpRefinement(
            sourceDown, targetDown,
            open3dFgrTransform,
            icp_distance,
            true,
            1e-6, 1e-6, 30,
            true
            );

        std::cout << "  [Custom ICP] Fitness: " << icpResult.fitness << std::endl;
        std::cout << "  [Custom ICP] RMSE:    " << icpResult.inlier_rmse << std::endl;
        std::cout << "  [Custom ICP] Transform:" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", icpResult.transformation[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        if (open3dIcpFitness >= 0.0f) {
            std::cout << "\n  --- Comparison: Custom ICP vs Open3D ICP (same FGR init) ---" << std::endl;
            std::cout << "    Custom  Fitness: " << icpResult.fitness
                      << "  RMSE: " << icpResult.inlier_rmse << std::endl;
            std::cout << "    Open3D  Fitness: " << open3dIcpFitness
                      << "  RMSE: " << open3dIcpRMSE << std::endl;

            float fitness_diff = std::abs(icpResult.fitness - open3dIcpFitness);
            float rmse_diff = std::abs(icpResult.inlier_rmse - open3dIcpRMSE);

            std::cout << "    Fitness diff: " << fitness_diff << std::endl;
            std::cout << "    RMSE diff:    " << rmse_diff << std::endl;

            float max_elem_diff = 0.0f;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    max_elem_diff = std::max(max_elem_diff,
                                             std::abs(icpResult.transformation[c][r] - open3dIcpTransform[c][r]));

            std::cout << "    Max matrix element diff: " << max_elem_diff << std::endl;

            std::cout << "\n    Custom ICP Transform:            Open3D ICP Transform:" << std::endl;
            for (int r = 0; r < 4; r++) {
                std::cout << "    [";
                for (int c = 0; c < 4; c++) {
                    char buf[16]; snprintf(buf, sizeof(buf), "%8.4f", icpResult.transformation[c][r]);
                    std::cout << buf;
                }
                std::cout << " ]  [";
                for (int c = 0; c < 4; c++) {
                    char buf[16]; snprintf(buf, sizeof(buf), "%8.4f", open3dIcpTransform[c][r]);
                    std::cout << buf;
                }
                std::cout << " ]" << std::endl;
            }

            if (fitness_diff < 0.01f && max_elem_diff < 0.05f) {
                std::cout << "     EXCELLENT: ICP results match Open3D closely" << std::endl;
            } else if (fitness_diff < 0.03f && max_elem_diff < 0.2f) {
                std::cout << "     ACCEPTABLE: ICP results reasonably close" << std::endl;
            } else {
                std::cout << "     DIVERGENT: ICP results differ significantly" << std::endl;
                if (fitness_diff > 0.05f)
                    std::cout << "       Fitness diff too large" << std::endl;
                if (max_elem_diff > 0.5f)
                    std::cout << "       Transform diff too large" << std::endl;
            }
        }
    }

    std::cout << "\n  --- Test B: Custom ICP + Custom FGR init (full pipeline) ---" << std::endl;
    {
        auto icpResult = reg.icpRefinement(
            sourceDown, targetDown,
            customFgrTransform,
            icp_distance,
            true
            );

        std::cout << "  [Custom ICP] Fitness: " << icpResult.fitness << std::endl;
        std::cout << "  [Custom ICP] RMSE:    " << icpResult.inlier_rmse << std::endl;
        std::cout << "  [Custom ICP] Transform:" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", icpResult.transformation[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        if (open3dIcpFitness >= 0.0f) {
            float fitness_diff = std::abs(icpResult.fitness - open3dIcpFitness);
            std::cout << "    vs Open3D ICP: fitness diff=" << fitness_diff << std::endl;
        }
    }

    if (o3dSourceDown && o3dTargetDown) {
        std::cout << "\n  --- Test C: Custom ICP + Open3D data + Open3D FGR init ---" << std::endl;
        std::cout << "  (Using Open3D's exact points and normals)" << std::endl;
        std::cout << "  O3D Source: " << o3dSourceDown->size() << " pts" << std::endl;
        std::cout << "  O3D Target: " << o3dTargetDown->size() << " pts, normals: "
                  << (o3dTargetDown->hasNormals() ? "YES" : "NO") << std::endl;

        {
            std::cout << "\n  === Data comparison (NN-matched) ===" << std::endl;

            {
                std::vector<glm::dvec3> o3dPtsD(o3dTargetDown->size());
                for (size_t i = 0; i < o3dTargetDown->size(); i++)
                    o3dPtsD[i] = glm::dvec3(o3dTargetDown->points[i]);
                NanoflannAdaptorD adaptorD(o3dPtsD);
                auto treeD = buildKDTreeD(adaptorD);

                int ptExact = 0;
                int ptClose = 0;
                int ptInVoxel = 0;
                double maxNNDist = 0.0;
                double sumNNDist = 0.0;

                int nrmMatch = 0;
                int nrmFlipped = 0;
                double maxNrmAngle = 0.0;

                for (size_t i = 0; i < targetDown->size(); i++) {
                    glm::dvec3 cp(targetDown->points[i]);
                    size_t nn_idx;
                    double nn_dist_sq;
                    searchKNN1D(*treeD, cp, nn_idx, nn_dist_sq);
                    double dist = std::sqrt(nn_dist_sq);

                    if (dist < 1e-6) ptExact++;
                    if (dist < 0.01) ptClose++;
                    if (dist < voxel_size * 0.5) ptInVoxel++;
                    maxNNDist = std::max(maxNNDist, dist);
                    sumNNDist += dist;

                    if (targetDown->hasNormals() && o3dTargetDown->hasNormals()
                        && dist < voxel_size) {
                        glm::dvec3 cn(targetDown->normals[i]);
                        glm::dvec3 on(o3dTargetDown->normals[nn_idx]);
                        double cn_len = glm::length(cn);
                        double on_len = glm::length(on);
                        if (cn_len > 1e-12 && on_len > 1e-12) {
                            double cosA = glm::dot(cn / cn_len, on / on_len);
                            if (cosA > 0.99) nrmMatch++;
                            else if (cosA < -0.99) nrmFlipped++;
                            double angle = std::acos(std::clamp(std::abs(cosA), 0.0, 1.0))
                                           * 180.0 / M_PI;
                            maxNrmAngle = std::max(maxNrmAngle, angle);
                        }
                    }
                }

                int n = static_cast<int>(targetDown->size());
                std::cout << "  Target (" << n << " vs " << o3dTargetDown->size() << " pts):" << std::endl;
                std::cout << "    Points exact (<1e-6):  " << ptExact << "/" << n << std::endl;
                std::cout << "    Points close (<0.01):  " << ptClose << "/" << n << std::endl;
                std::cout << "    Points in voxel (<" << voxel_size*0.5 << "): " << ptInVoxel << "/" << n << std::endl;
                std::cout << "    Max NN dist: " << maxNNDist << std::endl;
                std::cout << "    Mean NN dist: " << sumNNDist / n << std::endl;
                if (targetDown->hasNormals() && o3dTargetDown->hasNormals()) {
                    std::cout << "    Normals match (cos>0.99): " << nrmMatch << "/" << n << std::endl;
                    std::cout << "    Normals flipped (cos<-0.99): " << nrmFlipped << "/" << n << std::endl;
                    std::cout << "    Max normal angle diff: " << maxNrmAngle << " deg" << std::endl;
                }
            }

            {
                std::vector<glm::dvec3> o3dPtsD(o3dSourceDown->size());
                for (size_t i = 0; i < o3dSourceDown->size(); i++)
                    o3dPtsD[i] = glm::dvec3(o3dSourceDown->points[i]);
                NanoflannAdaptorD adaptorD(o3dPtsD);
                auto treeD = buildKDTreeD(adaptorD);

                int ptExact = 0, ptClose = 0, ptInVoxel = 0;
                double maxNNDist = 0.0, sumNNDist = 0.0;

                for (size_t i = 0; i < sourceDown->size(); i++) {
                    glm::dvec3 cp(sourceDown->points[i]);
                    size_t nn_idx;
                    double nn_dist_sq;
                    searchKNN1D(*treeD, cp, nn_idx, nn_dist_sq);
                    double dist = std::sqrt(nn_dist_sq);

                    if (dist < 1e-6) ptExact++;
                    if (dist < 0.01) ptClose++;
                    if (dist < voxel_size * 0.5) ptInVoxel++;
                    maxNNDist = std::max(maxNNDist, dist);
                    sumNNDist += dist;
                }

                int n = static_cast<int>(sourceDown->size());
                std::cout << "  Source (" << n << " vs " << o3dSourceDown->size() << " pts):" << std::endl;
                std::cout << "    Points exact (<1e-6):  " << ptExact << "/" << n << std::endl;
                std::cout << "    Points close (<0.01):  " << ptClose << "/" << n << std::endl;
                std::cout << "    Points in voxel (<" << voxel_size*0.5 << "): " << ptInVoxel << "/" << n << std::endl;
                std::cout << "    Max NN dist: " << maxNNDist << std::endl;
                std::cout << "    Mean NN dist: " << sumNNDist / n << std::endl;
            }
        }

        auto icpResultC = reg.icpRefinement(
            o3dSourceDown, o3dTargetDown,
            open3dFgrTransform,
            icp_distance,
            true,
            1e-6, 1e-6, 30,
            true
            );

        std::cout << "  [Custom ICP on O3D data] Fitness: " << icpResultC.fitness << std::endl;
        std::cout << "  [Custom ICP on O3D data] RMSE:    " << icpResultC.inlier_rmse << std::endl;
        std::cout << "  [Custom ICP on O3D data] Transform:" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", icpResultC.transformation[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        if (open3dIcpFitness >= 0.0f) {
            float fitness_diff = std::abs(icpResultC.fitness - open3dIcpFitness);
            float rmse_diff = std::abs(icpResultC.inlier_rmse - open3dIcpRMSE);
            std::cout << "\n  --- Test C vs Open3D ICP ---" << std::endl;
            std::cout << "    Custom(O3D data)  Fitness: " << icpResultC.fitness
                      << "  RMSE: " << icpResultC.inlier_rmse << std::endl;
            std::cout << "    Open3D ICP        Fitness: " << open3dIcpFitness
                      << "  RMSE: " << open3dIcpRMSE << std::endl;
            std::cout << "    Fitness diff: " << fitness_diff << std::endl;
            std::cout << "    RMSE diff:    " << rmse_diff << std::endl;

            if (fitness_diff < 0.01f) {
                std::cout << "     Algorithm MATCH: difference is in DATA, not algorithm" << std::endl;
            } else {
                std::cout << "     Algorithm MISMATCH: ICP solver differs from Open3D" << std::endl;
            }
        }
    } else {
        std::cout << "\n  (Test C skipped: Open3D point clouds not provided)" << std::endl;
    }

    std::cout << "\n  =====================================" << std::endl;
}

inline void debugStage8(
    const mCutMesh& screenMesh,
    int gridW, int gridH, float zThreshold,
    float voxel_size,
    std::vector<mCutMesh*>& organMeshes,
    std::vector<std::string>& meshNames,
    const glm::mat4& open3dIcpTransform,
    const glm::mat4& customIcpTransform,
    float open3dUmeyamaScale = -1.0f,
    int   open3dUmeyamaCorres = -1,
    float open3dUmeyamaAvgErr = -1.0f,
    const glm::mat4& open3dUmeyamaTransform = glm::mat4(1.0f),
    std::shared_ptr<PointCloud> o3dSourceDown = nullptr,
    std::shared_ptr<PointCloud> o3dTargetDown = nullptr)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Stage 8: Umeyama Refinement                  |" << std::endl;
    std::cout << "================================================" << std::endl;

    NoOpen3DRegistration reg;

    auto targetCloud = reg.extractFrontFacePoints(screenMesh, gridW, gridH, zThreshold);
    auto targetDown = reg.preprocess(targetCloud, voxel_size, false);

    auto sourceResult = buildSourcePointCloud(organMeshes, meshNames);
    auto sourceDown = reg.preprocess(sourceResult.cloud, voxel_size, true);

    float umeyama_threshold = voxel_size * 2.0f;

    std::cout << "  Source: " << sourceDown->size() << " points" << std::endl;
    std::cout << "  Target: " << targetDown->size() << " points" << std::endl;
    std::cout << "  Correspondence threshold: " << umeyama_threshold << std::endl;

    std::cout << "\n  --- Test A: Custom Umeyama + Open3D ICP init ---" << std::endl;
    {
        std::cout << "  Init transform (Open3D ICP):" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", open3dIcpTransform[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        auto umeyamaResult = reg.applyUmeyamaRefinement(
            sourceDown, targetDown, open3dIcpTransform, umeyama_threshold);

        std::cout << "\n  [Custom Umeyama] Scale: " << umeyamaResult.scale << std::endl;
        std::cout << "  [Custom Umeyama] Correspondences: " << umeyamaResult.correspondenceCount << std::endl;
        std::cout << "  [Custom Umeyama] Avg error: " << umeyamaResult.averageError << std::endl;
        std::cout << "  [Custom Umeyama] Success: " << (umeyamaResult.success ? "YES" : "NO") << std::endl;
        std::cout << "  [Custom Umeyama] Transform:" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", umeyamaResult.transformation[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        std::vector<glm::vec3> src_pts, tgt_pts;
        reg.extractCorrespondences(sourceDown, targetDown, open3dIcpTransform,
                                   umeyama_threshold, src_pts, tgt_pts);
        float scaleOnly = Reg3D::estimateScaleOnly(src_pts, tgt_pts);
        std::cout << "\n  [estimateScaleOnly] Scale: " << scaleOnly
                  << " (from " << src_pts.size() << " correspondences)" << std::endl;

        if (open3dUmeyamaScale >= 0.0f) {
            std::cout << "\n  --- Comparison: Custom Umeyama vs Open3D Umeyama (same ICP init) ---" << std::endl;
            std::cout << "    Custom  Scale: " << umeyamaResult.scale
                      << "  Corres: " << umeyamaResult.correspondenceCount
                      << "  AvgErr: " << umeyamaResult.averageError << std::endl;
            std::cout << "    Open3D  Scale: " << open3dUmeyamaScale
                      << "  Corres: " << open3dUmeyamaCorres
                      << "  AvgErr: " << open3dUmeyamaAvgErr << std::endl;

            float scale_diff = std::abs(umeyamaResult.scale - open3dUmeyamaScale);
            int corres_diff = std::abs(umeyamaResult.correspondenceCount - open3dUmeyamaCorres);
            float err_diff = std::abs(umeyamaResult.averageError - open3dUmeyamaAvgErr);

            std::cout << "    Scale diff: " << scale_diff << std::endl;
            std::cout << "    Corres diff: " << corres_diff << std::endl;
            std::cout << "    AvgErr diff: " << err_diff << std::endl;

            float max_t_diff = 0.0f;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) {
                    float d = std::abs(umeyamaResult.transformation[c][r] - open3dUmeyamaTransform[c][r]);
                    if (d > max_t_diff) max_t_diff = d;
                }
            std::cout << "    Max matrix element diff: " << max_t_diff << std::endl;

            std::cout << "\n    Custom Umeyama Transform:         Open3D Umeyama Transform:" << std::endl;
            for (int r = 0; r < 4; r++) {
                std::cout << "    [";
                for (int c = 0; c < 4; c++)
                    std::cout << std::fixed << std::setprecision(4)
                              << std::setw(9) << umeyamaResult.transformation[c][r];
                std::cout << " ]  [";
                for (int c = 0; c < 4; c++)
                    std::cout << std::fixed << std::setprecision(4)
                              << std::setw(9) << open3dUmeyamaTransform[c][r];
                std::cout << " ]" << std::endl;
            }

            if (scale_diff < 0.001f && corres_diff == 0 && max_t_diff < 0.001f) {
                std::cout << "     EXCELLENT: Umeyama results match Open3D exactly" << std::endl;
            } else if (scale_diff < 0.01f && max_t_diff < 0.01f) {
                std::cout << "     GOOD: Umeyama results closely match Open3D" << std::endl;
            } else if (scale_diff < 0.05f) {
                std::cout << "     ACCEPTABLE: Minor differences" << std::endl;
            } else {
                std::cout << "     MISMATCH: Significant differences" << std::endl;
            }
        } else {
            std::cout << "\n  (Open3D Umeyama results not provided, skipping comparison)" << std::endl;
        }
    }

    std::cout << "\n  --- Test B: Custom Umeyama + Custom ICP init (full pipeline) ---" << std::endl;
    {
        auto umeyamaResult = reg.applyUmeyamaRefinement(
            sourceDown, targetDown, customIcpTransform, umeyama_threshold);

        std::cout << "  [Custom Umeyama] Scale: " << umeyamaResult.scale << std::endl;
        std::cout << "  [Custom Umeyama] Correspondences: " << umeyamaResult.correspondenceCount << std::endl;
        std::cout << "  [Custom Umeyama] Avg error: " << umeyamaResult.averageError << std::endl;
        std::cout << "  [Custom Umeyama] Success: " << (umeyamaResult.success ? "YES" : "NO") << std::endl;
        std::cout << "  [Custom Umeyama] Transform:" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", umeyamaResult.transformation[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        if (open3dUmeyamaScale >= 0.0f) {
            float scale_diff = std::abs(umeyamaResult.scale - open3dUmeyamaScale);
            std::cout << "    vs Open3D scale diff: " << scale_diff << std::endl;
        }
    }

    if (o3dSourceDown && o3dTargetDown && !o3dSourceDown->empty()) {
        std::cout << "\n  --- Test C: Custom Umeyama + Open3D data + Open3D ICP init ---" << std::endl;
        std::cout << "  (Using Open3D's exact points)" << std::endl;
        std::cout << "  O3D Source: " << o3dSourceDown->size() << " pts" << std::endl;
        std::cout << "  O3D Target: " << o3dTargetDown->size() << " pts" << std::endl;

        {
            NanoflannAdaptor adaptA(targetDown->points);
            auto treeA = buildKDTree(adaptA);

            int ptExact = 0, ptClose = 0;
            float maxNNDist = 0.0f;
            for (size_t i = 0; i < o3dTargetDown->size(); i++) {
                size_t idx; float dsq;
                searchKNN1(*treeA, o3dTargetDown->points[i], idx, dsq);
                float dist = std::sqrt(dsq);
                if (dist < 1e-6f) ptExact++;
                if (dist < 0.01f) ptClose++;
                maxNNDist = std::max(maxNNDist, dist);
            }
            int n = static_cast<int>(o3dTargetDown->size());
            std::cout << "  Target points exact: " << ptExact << "/" << n
                      << ", close: " << ptClose << "/" << n
                      << ", max dist: " << maxNNDist << std::endl;
        }

        auto umeyamaResultC = reg.applyUmeyamaRefinement(
            o3dSourceDown, o3dTargetDown, open3dIcpTransform, umeyama_threshold);

        std::cout << "  [Custom Umeyama on O3D data] Scale: " << umeyamaResultC.scale << std::endl;
        std::cout << "  [Custom Umeyama on O3D data] Correspondences: " << umeyamaResultC.correspondenceCount << std::endl;
        std::cout << "  [Custom Umeyama on O3D data] Avg error: " << umeyamaResultC.averageError << std::endl;
        std::cout << "  [Custom Umeyama on O3D data] Transform:" << std::endl;
        for (int r = 0; r < 4; r++) {
            std::cout << "    [";
            for (int c = 0; c < 4; c++) {
                char buf[16]; snprintf(buf, sizeof(buf), "%9.4f", umeyamaResultC.transformation[c][r]);
                std::cout << buf;
            }
            std::cout << " ]" << std::endl;
        }

        if (open3dUmeyamaScale >= 0.0f) {
            float scale_diff = std::abs(umeyamaResultC.scale - open3dUmeyamaScale);
            int corres_diff = std::abs(umeyamaResultC.correspondenceCount - open3dUmeyamaCorres);
            float max_t_diff = 0.0f;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) {
                    float d = std::abs(umeyamaResultC.transformation[c][r] - open3dUmeyamaTransform[c][r]);
                    if (d > max_t_diff) max_t_diff = d;
                }

            std::cout << "\n  --- Test C vs Open3D Umeyama ---" << std::endl;
            std::cout << "    Custom(O3D data)  Scale: " << umeyamaResultC.scale
                      << "  Corres: " << umeyamaResultC.correspondenceCount << std::endl;
            std::cout << "    Open3D Umeyama    Scale: " << open3dUmeyamaScale
                      << "  Corres: " << open3dUmeyamaCorres << std::endl;
            std::cout << "    Scale diff: " << scale_diff << std::endl;
            std::cout << "    Corres diff: " << corres_diff << std::endl;
            std::cout << "    Max matrix diff: " << max_t_diff << std::endl;

            if (scale_diff < 0.001f && corres_diff == 0) {
                std::cout << "     Algorithm MATCH: difference is in DATA, not algorithm" << std::endl;
            } else if (scale_diff < 0.01f) {
                std::cout << "     GOOD: Close match" << std::endl;
            } else {
                std::cout << "     Algorithm MISMATCH" << std::endl;
            }
        }
    } else {
        std::cout << "\n  (Test C skipped: Open3D point clouds not provided)" << std::endl;
    }

    std::cout << "\n  =====================================" << std::endl;
}

inline bool performRegistrationMultiMeshWithScale(
    std::vector<mCutMesh*> organMeshes,
    std::vector<std::string> meshNames,
    mCutMesh* screenMesh,
    const glm::vec3& camera_position,
    int gridWidth = 128,
    int gridHeight = 72,
    int max_iterations = 15,
    float convergence_threshold = 0.005f,
    float min_fitness_for_convergence = 0.35f,
    bool estimate_scale = true,
    float min_scale_threshold = 0.03f,
    float zThreshold = 0.3f)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Custom Registration Pipeline               |" << std::endl;
    std::cout << "| (Open3D-Identical Logic)                   |" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "  Max iterations:       " << max_iterations << std::endl;
    std::cout << "  Convergence threshold:" << convergence_threshold << std::endl;
    std::cout << "  Min fitness required: " << min_fitness_for_convergence << std::endl;
    std::cout << "  Scale estimation:     " << (estimate_scale ? "ON" : "OFF") << std::endl;

    if (organMeshes.size() != meshNames.size()) {
        std::cerr << " Error: Mesh count and name count mismatch" << std::endl;
        return false;
    }

    NoOpen3DRegistration reg;

    float prev_fitness = 0.0f;
    float best_fitness = 0.0f;
    int best_iteration = 0;
    RegistrationResult final_result;

    std::vector<std::vector<float>> best_vertices(organMeshes.size());
    std::vector<std::vector<float>> best_normals(organMeshes.size());
    for (size_t m = 0; m < organMeshes.size(); m++) {
        best_vertices[m] = organMeshes[m]->mVertices;
        best_normals[m] = organMeshes[m]->mNormals;
    }

    int skipped_count = 0;
    int consecutive_skips = 0;

    {
        if (g_progressCallback) g_progressCallback(0.05f, "Full Auto: Building point clouds...");
        std::cout << "\n  Evaluating initial state..." << std::endl;
        auto targetCloud0 = reg.extractFrontFacePoints(*screenMesh, gridWidth, gridHeight, zThreshold);
        auto sourceResult0 = buildSourcePointCloud(organMeshes, meshNames);
        float voxel0 = 0.5f;
        auto srcDown0 = reg.preprocess(sourceResult0.cloud, voxel0, false);
        auto tgtDown0 = reg.preprocess(targetCloud0, voxel0, false);
        float eval_dist = voxel0 * 0.4f;
        auto evalResult = NoOpen3DRegistration::evaluateCurrentFitness(
            *srcDown0, *tgtDown0, eval_dist, glm::dmat4(1.0));
        prev_fitness = evalResult.fitness;
        best_fitness = evalResult.fitness;
        std::cout << "  Initial fitness: " << prev_fitness << std::endl;
    }

    for (int iter = 0; iter < max_iterations; iter++) {
        if (g_progressCallback) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Full Auto: Iteration %d / %d", iter + 1, max_iterations);
            g_progressCallback(0.1f + 0.85f * (float)iter / max_iterations, msg);
        }
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "  ITERATION " << (iter + 1) << "/" << max_iterations << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        std::vector<std::vector<float>> backup_vertices(organMeshes.size());
        std::vector<std::vector<float>> backup_normals(organMeshes.size());
        for (size_t m = 0; m < organMeshes.size(); m++) {
            backup_vertices[m] = organMeshes[m]->mVertices;
            backup_normals[m] = organMeshes[m]->mNormals;
        }

        try {
            std::cout << "Step 1: Converting and combining meshes to point cloud..." << std::endl;
            auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gridWidth, gridHeight, zThreshold);
            auto sourceResult = buildSourcePointCloud(organMeshes, meshNames);

            std::cout << "  Total combined surface points: " << sourceResult.cloud->size() << std::endl;

            if (targetCloud->size() < 100 || sourceResult.cloud->size() < 100) {
                std::cerr << " Error: Not enough points" << std::endl;
                return false;
            }

            float voxel_size = 0.5f;
            std::cout << "\nStep 2: Preprocessing (voxel size: " << voxel_size << ")..." << std::endl;

            auto targetDown = reg.preprocess(targetCloud, voxel_size, false);
            auto sourceDown = reg.preprocess(sourceResult.cloud, voxel_size, true);

            std::cout << "  Screen points after downsampling: " << targetDown->size() << std::endl;
            std::cout << "  Organ points after downsampling: " << sourceDown->size() << std::endl;

            if (targetDown->size() < 10 || sourceDown->size() < 10) {
                std::cerr << " Error: Too few points after downsampling" << std::endl;
                return false;
            }

            std::cout << "\nStep 3: Computing FPFH features..." << std::endl;
            auto targetFpfh = reg.computeFPFH(targetDown, voxel_size);
            auto sourceFpfh = reg.computeFPFH(sourceDown, voxel_size);

            std::cout << "\nStep 4: Fast Global Registration..." << std::endl;

            auto result = reg.fastGlobalRegistration(sourceDown, targetDown, sourceFpfh, targetFpfh, voxel_size);

            std::cout << "  Initial fitness: " << result.fitness << std::endl;
            std::cout << "  Initial RMSE: " << result.inlier_rmse << std::endl;

            auto sourceDown_std = sourceDown;
            auto targetDown_std = targetDown;
            float standard_voxel = 0.5f;

            if (result.fitness < 0.3f) {
                std::cout << "\n  Low fitness score. Retrying with adjusted parameters..." << std::endl;

                float retry_voxel = voxel_size * 1.2f;
                auto targetDown_retry = reg.preprocess(targetCloud, retry_voxel, false);
                auto sourceDown_retry = reg.preprocess(sourceResult.cloud, retry_voxel, true);

                std::cout << "  Retry downsampled: screen=" << targetDown_retry->size()
                          << ", organ=" << sourceDown_retry->size() << std::endl;

                if (targetDown_retry->size() >= 100 && sourceDown_retry->size() >= 100) {
                    try {
                        auto targetFpfh_retry = reg.computeFPFH(targetDown_retry, retry_voxel);
                        auto sourceFpfh_retry = reg.computeFPFH(sourceDown_retry, retry_voxel);

                        auto result_retry = reg.fastGlobalRegistration(
                            sourceDown_retry, targetDown_retry,
                            sourceFpfh_retry, targetFpfh_retry,
                            retry_voxel);

                        std::cout << "  Retry fitness: " << result_retry.fitness << std::endl;
                        std::cout << "  Retry RMSE: " << result_retry.inlier_rmse << std::endl;

                        if (result_retry.fitness > result.fitness) {
                            result = result_retry;
                            voxel_size = retry_voxel;
                            targetDown = targetDown_retry;
                            sourceDown = sourceDown_retry;
                            std::cout << "  Using retry result" << std::endl;
                        } else {
                            std::cout << "  Retry did not improve, using original result" << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "  Retry failed: " << e.what() << std::endl;
                        std::cout << "  Using original result" << std::endl;
                    }
                } else {
                    std::cout << "  Insufficient points for retry. Using original result." << std::endl;
                }
            }

            std::cout << "\nStep 5: ICP refinement..." << std::endl;
            float icp_distance = voxel_size * 0.4f;

            try {
                result = reg.icpRefinement(sourceDown, targetDown, result.transformation, icp_distance, true);
                std::cout << "  Final fitness: " << result.fitness << std::endl;
                std::cout << "  Final RMSE: " << result.inlier_rmse << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  ICP refinement failed: " << e.what() << std::endl;
                std::cout << "  Continuing with FGR result" << std::endl;
            }

            if (voxel_size != standard_voxel) {
                float standard_dist = standard_voxel * 0.4f;
                auto evalResult = NoOpen3DRegistration::evaluateCurrentFitness(
                    *sourceDown_std, *targetDown_std, standard_dist, result.transformation);
                std::cout << "  Re-evaluated fitness (standard): " << evalResult.fitness << std::endl;
                std::cout << "  Re-evaluated RMSE (standard): " << evalResult.inlier_rmse << std::endl;
                result.fitness = evalResult.fitness;
                result.inlier_rmse = evalResult.inlier_rmse;
            }

            final_result = result;

            float current_fitness = result.fitness;
            bool should_apply_transform = true;

            if (current_fitness < prev_fitness - 0.05f) {
                std::cout << "\n  Quality degradation detected!" << std::endl;
                std::cout << "  Previous fitness: " << prev_fitness << std::endl;
                std::cout << "  Current fitness:  " << current_fitness << std::endl;
                std::cout << "  Skipping this transformation" << std::endl;
                should_apply_transform = false;
                skipped_count++;
                consecutive_skips++;

                if (consecutive_skips >= 3) {
                    std::cout << "\n  3 consecutive skips detected. Registration may have converged." << std::endl;
                    break;
                }
            }

            if (should_apply_transform) {
                consecutive_skips = 0;

                glm::mat4 glm_transform;
                glm::mat4 rigid_transform = result.transformation;
                float current_scale = 1.0f;

                if (estimate_scale) {
                    std::cout << "\nStep 5.5: Scale estimation..." << std::endl;

                    std::vector<glm::vec3> src_pts, tgt_pts;
                    reg.extractCorrespondences(sourceDown, targetDown, result.transformation,
                                               voxel_size * 2.0f, src_pts, tgt_pts);

                    std::cout << "  Found " << src_pts.size() << " correspondences" << std::endl;

                    if (src_pts.size() >= 10) {
                        current_scale = Reg3D::estimateScaleOnly(src_pts, tgt_pts);
                        std::cout << "  Estimated scale: " << current_scale << std::endl;

                        if (current_scale > 0.5f && current_scale < 1.5f) {
                            if (std::abs(current_scale - 1.0f) > min_scale_threshold) {
                                std::cout << "  Applying scale correction" << std::endl;

                                glm::mat4 scale_matrix = glm::mat4(1.0f);
                                scale_matrix[0][0] = current_scale;
                                scale_matrix[1][1] = current_scale;
                                scale_matrix[2][2] = current_scale;

                                glm_transform = scale_matrix * rigid_transform;
                            } else {
                                std::cout << "  Scale ~ 1.0, no correction needed" << std::endl;
                                glm_transform = rigid_transform;
                                current_scale = 1.0f;
                            }
                        } else {
                            std::cout << "  Scale out of valid range, ignoring" << std::endl;
                            glm_transform = rigid_transform;
                            current_scale = 1.0f;
                        }
                    } else {
                        std::cout << "  Not enough correspondences for scale estimation" << std::endl;
                        glm_transform = rigid_transform;
                    }
                } else {
                    glm_transform = rigid_transform;
                }

                std::cout << "\nStep 6: Applying transformation..." << std::endl;

                if (iter == 0) {
                    std::cout << "  Transformation matrix:" << std::endl;
                    for (int i = 0; i < 4; i++) {
                        std::cout << "  [";
                        for (int j = 0; j < 4; j++) {
                            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << glm_transform[j][i];
                            if (j < 3) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                    }
                }

                for (size_t m = 0; m < organMeshes.size(); m++) {
                    for (size_t i = 0; i < organMeshes[m]->mVertices.size(); i += 3) {
                        glm::vec4 vertex(
                            organMeshes[m]->mVertices[i],
                            organMeshes[m]->mVertices[i + 1],
                            organMeshes[m]->mVertices[i + 2],
                            1.0f);
                        vertex = glm_transform * vertex;
                        organMeshes[m]->mVertices[i]     = vertex.x;
                        organMeshes[m]->mVertices[i + 1] = vertex.y;
                        organMeshes[m]->mVertices[i + 2] = vertex.z;
                    }

                    if (!organMeshes[m]->mNormals.empty()) {
                        glm::mat3 rotation_matrix = glm::mat3(rigid_transform);
                        for (size_t i = 0; i < organMeshes[m]->mNormals.size(); i += 3) {
                            glm::vec3 normal(
                                organMeshes[m]->mNormals[i],
                                organMeshes[m]->mNormals[i + 1],
                                organMeshes[m]->mNormals[i + 2]);
                            normal = glm::normalize(rotation_matrix * normal);
                            organMeshes[m]->mNormals[i]     = normal.x;
                            organMeshes[m]->mNormals[i + 1] = normal.y;
                            organMeshes[m]->mNormals[i + 2] = normal.z;
                        }
                    }

                    setUp(*organMeshes[m]);
                }

                if (current_fitness > best_fitness) {
                    best_fitness = current_fitness;
                    best_iteration = iter + 1;
                    for (size_t m = 0; m < organMeshes.size(); m++) {
                        best_vertices[m] = organMeshes[m]->mVertices;
                        best_normals[m] = organMeshes[m]->mNormals;
                    }
                    std::cout << "  New best fitness: " << best_fitness << std::endl;
                }

                prev_fitness = current_fitness;

            } else {
                for (size_t m = 0; m < organMeshes.size(); m++) {
                    organMeshes[m]->mVertices = backup_vertices[m];
                    organMeshes[m]->mNormals = backup_normals[m];
                    setUp(*organMeshes[m]);
                }
            }

            float fitness_change = std::abs(result.fitness - prev_fitness);

            std::cout << "\n  Summary:" << std::endl;
            std::cout << "  - Fitness: " << result.fitness;
            if (should_apply_transform) {
                std::cout << " (change: " << fitness_change << ")" << std::endl;
            } else {
                std::cout << " [SKIPPED]" << std::endl;
            }
            std::cout << "  - Best fitness so far: " << best_fitness << " (iter " << best_iteration << ")" << std::endl;
            std::cout << "  - Skipped count: " << skipped_count << "/" << (iter + 1) << std::endl;

            if (iter >= 3 && should_apply_transform) {
                if (result.fitness >= min_fitness_for_convergence && fitness_change < convergence_threshold) {
                    std::cout << "\n Converged!" << std::endl;
                    break;
                }
                if (result.fitness > 0.50f) {
                    std::cout << "\n Excellent fitness achieved!" << std::endl;
                    break;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "\n Exception in iteration " << (iter+1) << ": " << e.what() << std::endl;

            for (size_t m = 0; m < organMeshes.size(); m++) {
                organMeshes[m]->mVertices = best_vertices[m];
                organMeshes[m]->mNormals = best_normals[m];
                setUp(*organMeshes[m]);
            }

            if (iter > 0) {
                std::cout << " Rolled back to best state (iter " << best_iteration << ")" << std::endl;
                break;
            } else {
                return false;
            }
        }
    }

    std::cout << "\n" << std::endl;
    if (best_iteration == 0) {
        std::cout << "  Keeping initial state (no iteration improved upon it, fitness=" << best_fitness << ")" << std::endl;
    } else {
        std::cout << "  Applying best result (iter " << best_iteration << ", fitness " << best_fitness << ")" << std::endl;
    }
    for (size_t m = 0; m < organMeshes.size(); m++) {
        organMeshes[m]->mVertices = best_vertices[m];
        organMeshes[m]->mNormals = best_normals[m];
        setUp(*organMeshes[m]);
    }

    std::cout << "\nStep 7: Extracting and sampling correspondences..." << std::endl;

    try {
        auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gridWidth, gridHeight, zThreshold);
        auto sourceResult = buildSourcePointCloud(organMeshes, meshNames);

        std::vector<glm::vec3> source_points, target_points;
        reg.extractCorrespondences(
            sourceResult.cloud, targetCloud,
            glm::mat4(1.0f),
            1.0f,
            source_points, target_points);

        std::cout << "  Found " << source_points.size() << " correspondences" << std::endl;

        int max_markers = Reg3D::g_maxCorrespondenceMarkers;
        std::vector<glm::vec3> sampled_source, sampled_target;

        if (source_points.size() > static_cast<size_t>(max_markers)) {
            float step = static_cast<float>(source_points.size()) / max_markers;
            for (int i = 0; i < max_markers; i++) {
                int idx = static_cast<int>(i * step);
                if (idx < static_cast<int>(source_points.size())) {
                    sampled_source.push_back(source_points[idx]);
                    sampled_target.push_back(target_points[idx]);
                }
            }
            std::cout << "  Sampled to " << sampled_source.size() << " markers" << std::endl;
        } else {
            sampled_source = source_points;
            sampled_target = target_points;
        }

        registrationHandle.objectPoints.clear();
        registrationHandle.boardPoints.clear();
        for (const auto& sp : sampled_source)
            registrationHandle.objectPoints.push_back(sp);
        for (const auto& tp : sampled_target)
            registrationHandle.boardPoints.push_back(tp);

        {
            float totalErr = 0.0f;
            float sumSq = 0.0f;
            float maxErr = 0.0f;
            for (size_t i = 0; i < sampled_source.size(); i++) {
                float d = glm::distance(sampled_source[i], sampled_target[i]);
                totalErr += d;
                sumSq += d * d;
                if (d > maxErr) maxErr = d;
            }
            float n = sampled_source.empty() ? 1.0f : (float)sampled_source.size();
            registrationHandle.averageError = totalErr / n;
            registrationHandle.rmse = std::sqrt(sumSq / n);
            registrationHandle.maxError = maxErr;
        }
        registrationHandle.scaleFactor = Reg3D::estimateScaleOnly(
            registrationHandle.objectPoints, registrationHandle.boardPoints);

        registrationHandle.fitness = best_fitness;
        registrationHandle.icpRmse = final_result.inlier_rmse;
        registrationHandle.bestIteration = best_iteration;
        registrationHandle.refineCount = 0;

        registrationHandle.useRegistration = true;
        registrationHandle.state = RegistrationData::REGISTERED;

    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not extract correspondences: " << e.what() << std::endl;
    }

    if (g_progressCallback) g_progressCallback(1.0f, "Full Auto: Complete!");

    std::cout << "\n================================================" << std::endl;
    std::cout << "| [OK] Registration Complete                 |" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "  Best fitness:         " << best_fitness << " (iteration " << best_iteration << ")" << std::endl;
    std::cout << "  Final RMSE:           " << final_result.inlier_rmse << std::endl;
    std::cout << "  Avg Error:            " << registrationHandle.averageError << std::endl;
    std::cout << "  RMSE:                 " << registrationHandle.rmse << std::endl;
    std::cout << "  Max Error:            " << registrationHandle.maxError << std::endl;
    std::cout << "  Scale estimation:     " << (estimate_scale ? "Used" : "Not used") << std::endl;
    std::cout << "  Skipped iterations:   " << skipped_count << std::endl;
    std::cout << "  Meshes transformed:   " << organMeshes.size() << std::endl;

    return true;
}

struct VisibilityResultCustom {
    std::shared_ptr<PointCloud> cloud;
    std::vector<glm::vec3> points;
    std::vector<size_t> vertexIndices;
    int visibleCount = 0;
    int occludedCount = 0;
    int backfaceCount = 0;
    int totalVertices = 0;
};

inline VisibilityResultCustom extractVisibleVerticesCustom(
    const mCutMesh& mesh,
    const Reg3D::BVHTree& bvhTree,
    const glm::vec3& cameraPos,
    const glm::vec3& cameraTarget)
{
    VisibilityResultCustom result;
    result.cloud = std::make_shared<PointCloud>();

    glm::vec3 viewDirection = glm::normalize(cameraTarget - cameraPos);

    std::cout << "  Camera position: (" << cameraPos.x << ", "
              << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
    std::cout << "  View direction: (" << viewDirection.x << ", "
              << viewDirection.y << ", " << viewDirection.z << ")" << std::endl;

    int vertexCount = mesh.mVertices.size() / 3;
    result.totalVertices = vertexCount;

    const float EPSILON = 0.01f;

    for (int i = 0; i < vertexCount; i++) {
        glm::vec3 vertex(
            mesh.mVertices[i * 3],
            mesh.mVertices[i * 3 + 1],
            mesh.mVertices[i * 3 + 2]);

        glm::vec3 rayDir = glm::normalize(vertex - cameraPos);
        float distToVertex = glm::length(vertex - cameraPos);

        float dotProduct = glm::dot(rayDir, viewDirection);
        if (dotProduct < 0.3f) {
            result.backfaceCount++;
            continue;
        }

        float closestHit = FLT_MAX;
        for (const auto& tri : bvhTree.triangles) {
            glm::vec3 edge1 = tri.v1 - tri.v0;
            glm::vec3 edge2 = tri.v2 - tri.v0;
            glm::vec3 h = glm::cross(rayDir, edge2);
            float a = glm::dot(edge1, h);

            if (std::abs(a) < 1e-8f) continue;

            float f = 1.0f / a;
            glm::vec3 s = cameraPos - tri.v0;
            float u = f * glm::dot(s, h);
            if (u < 0.0f || u > 1.0f) continue;

            glm::vec3 q = glm::cross(s, edge1);
            float v = f * glm::dot(rayDir, q);
            if (v < 0.0f || u + v > 1.0f) continue;

            float t = f * glm::dot(edge2, q);
            if (t > EPSILON && t < closestHit) {
                closestHit = t;
            }
        }

        if (std::abs(closestHit - distToVertex) < distToVertex * 0.05f) {
            result.visibleCount++;
            result.vertexIndices.push_back(static_cast<size_t>(i));
            result.points.push_back(vertex);

            if (!mesh.mNormals.empty()) {
                result.cloud->addPointWithNormal(vertex, glm::vec3(
                                                             mesh.mNormals[i * 3],
                                                             mesh.mNormals[i * 3 + 1],
                                                             mesh.mNormals[i * 3 + 2]));
            } else {
                result.cloud->addPoint(vertex);
            }
        } else {
            result.occludedCount++;
        }
    }

    std::cout << "  Total vertices: " << result.totalVertices << std::endl;
    std::cout << "  Visible: " << result.visibleCount << std::endl;
    std::cout << "  Occluded: " << result.occludedCount << std::endl;
    std::cout << "  Backface: " << result.backfaceCount << std::endl;

    return result;
}

inline glm::vec3 computeTargetViewDirectionCustom(
    std::shared_ptr<PointCloud> targetCloud)
{
    if (targetCloud->empty()) {
        return glm::vec3(0.0f, 0.0f, -1.0f);
    }

    glm::vec3 centroid(0.0f);
    for (const auto& pt : targetCloud->points) {
        centroid += pt;
    }
    centroid /= static_cast<float>(targetCloud->size());

    if (targetCloud->hasNormals()) {
        glm::vec3 avgNormal(0.0f);
        for (const auto& n : targetCloud->normals) {
            avgNormal += n;
        }
        avgNormal = glm::normalize(avgNormal);
        return glm::normalize(-avgNormal);
    } else {
        float maxDist = 0.0f;
        glm::vec3 farthestPoint = centroid;
        for (const auto& pt : targetCloud->points) {
            float dist = glm::length(pt - centroid);
            if (dist > maxDist) {
                maxDist = dist;
                farthestPoint = pt;
            }
        }
        return glm::normalize(centroid - farthestPoint);
    }
}

inline std::vector<Reg3D::VertexCluster> selectTop2ClustersCustom(
    const Reg3D::RaycastClusteringResult& clusteringResult,
    std::shared_ptr<PointCloud> targetCloud)
{
    std::cout << "\n========== Custom Top-2 Cluster Selection ==========\n";

    glm::vec3 targetViewDir = computeTargetViewDirectionCustom(targetCloud);
    std::cout << "Target View Direction: (" << targetViewDir.x << ", "
              << targetViewDir.y << ", " << targetViewDir.z << ")\n";

    struct ClusterSim {
        int clusterId;
        float similarity;
        bool operator<(const ClusterSim& o) const { return similarity > o.similarity; }
    };

    std::vector<ClusterSim> similarities;
    for (const auto& cluster : clusteringResult.clusters) {
        float similarity = glm::dot(targetViewDir, -cluster.viewDirection);
        similarities.push_back({cluster.clusterId, similarity});
        std::cout << "Cluster " << cluster.clusterId
                  << ": Similarity = " << similarity << "\n";
    }

    std::sort(similarities.begin(), similarities.end());

    std::vector<Reg3D::VertexCluster> selected;
    std::cout << "\nSelected Top-2 Clusters:\n";
    for (int i = 0; i < std::min(2, static_cast<int>(similarities.size())); i++) {
        int cid = similarities[i].clusterId;
        for (const auto& cluster : clusteringResult.clusters) {
            if (cluster.clusterId == cid) {
                selected.push_back(cluster);
                std::cout << "  Rank " << (i+1) << ": Cluster " << cid
                          << " (Similarity: " << similarities[i].similarity
                          << ", Vertices: " << cluster.visibleVertices.size() << ")\n";
                break;
            }
        }
    }
    std::cout << "==========================================\n";

    return selected;
}

inline bool performRegistrationSingleMesh(
    std::vector<mCutMesh*> organMeshes,
    mCutMesh* sourceMesh,
    const std::vector<size_t>& sourceVertexIndices,
    mCutMesh* screenMesh,
    const glm::vec3& camera_position,
    int gridWidth = 128,
    int gridHeight = 72,
    int max_iterations = 15,
    float convergence_threshold = 0.005f,
    float min_fitness_for_convergence = 0.35f,
    bool estimate_scale = true,
    float min_scale_threshold = 0.03f,
    float zThreshold = 0.3f)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "| Custom Registration (Single Mesh Source)    |" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "  Source vertices:      " << sourceVertexIndices.size() << std::endl;
    std::cout << "  Max iterations:       " << max_iterations << std::endl;
    std::cout << "  Convergence threshold:" << convergence_threshold << std::endl;
    std::cout << "  Min fitness required: " << min_fitness_for_convergence << std::endl;
    std::cout << "  Scale estimation:     " << (estimate_scale ? "ON" : "OFF") << std::endl;

    if (sourceVertexIndices.size() < 10) {
        std::cerr << " Error: Not enough source vertices" << std::endl;
        return false;
    }

    NoOpen3DRegistration reg;

    float prev_fitness = 0.0f;
    float best_fitness = 0.0f;
    int best_iteration = 0;
    RegistrationResult final_result;

    std::vector<std::vector<float>> best_vertices(organMeshes.size());
    std::vector<std::vector<float>> best_normals(organMeshes.size());
    for (size_t m = 0; m < organMeshes.size(); m++) {
        best_vertices[m] = organMeshes[m]->mVertices;
        best_normals[m] = organMeshes[m]->mNormals;
    }

    int skipped_count = 0;
    int consecutive_skips = 0;

    {
        if (g_progressCallback) g_progressCallback(0.05f, "Hemi Auto: Building point clouds...");
        std::cout << "\n  Evaluating initial state..." << std::endl;
        auto targetCloud0 = reg.extractFrontFacePoints(*screenMesh, gridWidth, gridHeight, zThreshold);
        auto sourceCloud0 = std::make_shared<PointCloud>();
        for (size_t idx : sourceVertexIndices) {
            if (idx * 3 + 2 < sourceMesh->mVertices.size()) {
                sourceCloud0->addPoint(glm::vec3(
                    sourceMesh->mVertices[idx * 3],
                    sourceMesh->mVertices[idx * 3 + 1],
                    sourceMesh->mVertices[idx * 3 + 2]));
            }
        }
        float voxel0 = 0.5f;
        auto srcDown0 = reg.preprocess(sourceCloud0, voxel0, false);
        auto tgtDown0 = reg.preprocess(targetCloud0, voxel0, false);
        float eval_dist = voxel0 * 0.4f;
        auto evalResult = NoOpen3DRegistration::evaluateCurrentFitness(
            *srcDown0, *tgtDown0, eval_dist, glm::dmat4(1.0));
        prev_fitness = evalResult.fitness;
        best_fitness = evalResult.fitness;
        std::cout << "  Initial fitness: " << prev_fitness << std::endl;
    }

    for (int iter = 0; iter < max_iterations; iter++) {
        if (g_progressCallback) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Hemi Auto: Iteration %d / %d", iter + 1, max_iterations);
            g_progressCallback(0.1f + 0.85f * (float)iter / max_iterations, msg);
        }
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "  ITERATION " << (iter + 1) << "/" << max_iterations << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        std::vector<std::vector<float>> backup_vertices(organMeshes.size());
        std::vector<std::vector<float>> backup_normals(organMeshes.size());
        for (size_t m = 0; m < organMeshes.size(); m++) {
            backup_vertices[m] = organMeshes[m]->mVertices;
            backup_normals[m] = organMeshes[m]->mNormals;
        }

        try {
            std::cout << "Step 1: Building point clouds..." << std::endl;

            auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gridWidth, gridHeight, zThreshold);

            auto sourceCloud = std::make_shared<PointCloud>();
            for (size_t idx : sourceVertexIndices) {
                if (idx * 3 + 2 < sourceMesh->mVertices.size()) {
                    glm::vec3 pos(
                        sourceMesh->mVertices[idx * 3],
                        sourceMesh->mVertices[idx * 3 + 1],
                        sourceMesh->mVertices[idx * 3 + 2]);
                    if (!sourceMesh->mNormals.empty() && idx * 3 + 2 < sourceMesh->mNormals.size()) {
                        glm::vec3 nrm(
                            sourceMesh->mNormals[idx * 3],
                            sourceMesh->mNormals[idx * 3 + 1],
                            sourceMesh->mNormals[idx * 3 + 2]);
                        sourceCloud->addPointWithNormal(pos, nrm);
                    } else {
                        sourceCloud->addPoint(pos);
                    }
                }
            }

            std::cout << "  Source points: " << sourceCloud->size() << std::endl;
            std::cout << "  Target points: " << targetCloud->size() << std::endl;

            if (targetCloud->size() < 100 || sourceCloud->size() < 50) {
                std::cerr << " Error: Not enough points" << std::endl;
                return false;
            }

            float voxel_size = 0.5f;
            std::cout << "\nStep 2: Preprocessing (voxel size: " << voxel_size << ")..." << std::endl;

            auto targetDown = reg.preprocess(targetCloud, voxel_size, false);
            auto sourceDown = reg.preprocess(sourceCloud, voxel_size, true);

            std::cout << "  Screen points after downsampling: " << targetDown->size() << std::endl;
            std::cout << "  Source points after downsampling: " << sourceDown->size() << std::endl;

            if (targetDown->size() < 10 || sourceDown->size() < 10) {
                std::cerr << " Error: Too few points after downsampling" << std::endl;
                return false;
            }

            std::cout << "\nStep 3: Computing FPFH features..." << std::endl;
            auto targetFpfh = reg.computeFPFH(targetDown, voxel_size);
            auto sourceFpfh = reg.computeFPFH(sourceDown, voxel_size);

            std::cout << "\nStep 4: Fast Global Registration..." << std::endl;

            auto result = reg.fastGlobalRegistration(sourceDown, targetDown, sourceFpfh, targetFpfh, voxel_size);

            std::cout << "  Initial fitness: " << result.fitness << std::endl;
            std::cout << "  Initial RMSE: " << result.inlier_rmse << std::endl;

            auto sourceDown_std = sourceDown;
            auto targetDown_std = targetDown;
            float standard_voxel = 0.5f;

            if (result.fitness < 0.3f) {
                std::cout << "\n  Low fitness score. Retrying..." << std::endl;

                float retry_voxel = voxel_size * 1.2f;
                auto targetDown_retry = reg.preprocess(targetCloud, retry_voxel, false);
                auto sourceDown_retry = reg.preprocess(sourceCloud, retry_voxel, true);

                if (targetDown_retry->size() >= 100 && sourceDown_retry->size() >= 100) {
                    try {
                        auto targetFpfh_retry = reg.computeFPFH(targetDown_retry, retry_voxel);
                        auto sourceFpfh_retry = reg.computeFPFH(sourceDown_retry, retry_voxel);

                        auto result_retry = reg.fastGlobalRegistration(
                            sourceDown_retry, targetDown_retry,
                            sourceFpfh_retry, targetFpfh_retry, retry_voxel);

                        std::cout << "  Retry fitness: " << result_retry.fitness << std::endl;

                        if (result_retry.fitness > result.fitness) {
                            result = result_retry;
                            voxel_size = retry_voxel;
                            targetDown = targetDown_retry;
                            sourceDown = sourceDown_retry;
                            std::cout << "  Using retry result" << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "  Retry failed: " << e.what() << std::endl;
                    }
                }
            }

            std::cout << "\nStep 5: ICP refinement..." << std::endl;
            float icp_distance = voxel_size * 0.4f;

            try {
                result = reg.icpRefinement(sourceDown, targetDown, result.transformation, icp_distance, true);
                std::cout << "  Final fitness: " << result.fitness << std::endl;
                std::cout << "  Final RMSE: " << result.inlier_rmse << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  ICP failed: " << e.what() << std::endl;
            }

            if (voxel_size != standard_voxel) {
                float standard_dist = standard_voxel * 0.4f;
                auto evalResult = NoOpen3DRegistration::evaluateCurrentFitness(
                    *sourceDown_std, *targetDown_std, standard_dist, result.transformation);
                std::cout << "  Re-evaluated fitness (standard): " << evalResult.fitness << std::endl;
                std::cout << "  Re-evaluated RMSE (standard): " << evalResult.inlier_rmse << std::endl;
                result.fitness = evalResult.fitness;
                result.inlier_rmse = evalResult.inlier_rmse;
            }

            final_result = result;

            float current_fitness = result.fitness;
            bool should_apply_transform = true;

            if (current_fitness < prev_fitness - 0.05f) {
                std::cout << "\n  Quality degradation detected!" << std::endl;
                std::cout << "  Previous fitness: " << prev_fitness << std::endl;
                std::cout << "  Current fitness:  " << current_fitness << std::endl;
                std::cout << "  Skipping this transformation" << std::endl;
                should_apply_transform = false;
                skipped_count++;
                consecutive_skips++;

                if (consecutive_skips >= 3) {
                    std::cout << "\n  3 consecutive skips." << std::endl;
                    break;
                }
            }

            if (should_apply_transform) {
                consecutive_skips = 0;

                glm::mat4 glm_transform;
                glm::mat4 rigid_transform = result.transformation;
                float current_scale = 1.0f;

                if (estimate_scale) {
                    std::cout << "\nStep 5.5: Scale estimation..." << std::endl;

                    std::vector<glm::vec3> src_pts, tgt_pts;
                    reg.extractCorrespondences(sourceDown, targetDown, result.transformation,
                                               voxel_size * 2.0f, src_pts, tgt_pts);

                    std::cout << "  Found " << src_pts.size() << " correspondences" << std::endl;

                    if (src_pts.size() >= 10) {
                        current_scale = Reg3D::estimateScaleOnly(src_pts, tgt_pts);
                        std::cout << "  Estimated scale: " << current_scale << std::endl;

                        if (current_scale > 0.5f && current_scale < 1.5f) {
                            if (std::abs(current_scale - 1.0f) > min_scale_threshold) {
                                std::cout << "  Applying scale correction" << std::endl;
                                glm::mat4 scale_matrix = glm::mat4(1.0f);
                                scale_matrix[0][0] = current_scale;
                                scale_matrix[1][1] = current_scale;
                                scale_matrix[2][2] = current_scale;
                                glm_transform = scale_matrix * rigid_transform;
                            } else {
                                std::cout << "  Scale ~ 1.0, no correction needed" << std::endl;
                                glm_transform = rigid_transform;
                                current_scale = 1.0f;
                            }
                        } else {
                            std::cout << "  Scale out of valid range, ignoring" << std::endl;
                            glm_transform = rigid_transform;
                            current_scale = 1.0f;
                        }
                    } else {
                        std::cout << "  Not enough correspondences for scale" << std::endl;
                        glm_transform = rigid_transform;
                    }
                } else {
                    glm_transform = rigid_transform;
                }

                std::cout << "\nStep 6: Applying transformation..." << std::endl;

                if (iter == 0) {
                    std::cout << "  Transformation matrix:" << std::endl;
                    for (int i = 0; i < 4; i++) {
                        std::cout << "  [";
                        for (int j = 0; j < 4; j++) {
                            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << glm_transform[j][i];
                            if (j < 3) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                    }
                }

                for (size_t m = 0; m < organMeshes.size(); m++) {
                    for (size_t i = 0; i < organMeshes[m]->mVertices.size(); i += 3) {
                        glm::vec4 vertex(
                            organMeshes[m]->mVertices[i],
                            organMeshes[m]->mVertices[i + 1],
                            organMeshes[m]->mVertices[i + 2],
                            1.0f);
                        vertex = glm_transform * vertex;
                        organMeshes[m]->mVertices[i]     = vertex.x;
                        organMeshes[m]->mVertices[i + 1] = vertex.y;
                        organMeshes[m]->mVertices[i + 2] = vertex.z;
                    }

                    if (!organMeshes[m]->mNormals.empty()) {
                        glm::mat3 rotation_matrix = glm::mat3(rigid_transform);
                        for (size_t i = 0; i < organMeshes[m]->mNormals.size(); i += 3) {
                            glm::vec3 normal(
                                organMeshes[m]->mNormals[i],
                                organMeshes[m]->mNormals[i + 1],
                                organMeshes[m]->mNormals[i + 2]);
                            normal = glm::normalize(rotation_matrix * normal);
                            organMeshes[m]->mNormals[i]     = normal.x;
                            organMeshes[m]->mNormals[i + 1] = normal.y;
                            organMeshes[m]->mNormals[i + 2] = normal.z;
                        }
                    }

                    setUp(*organMeshes[m]);
                }

                if (current_fitness > best_fitness) {
                    best_fitness = current_fitness;
                    best_iteration = iter + 1;
                    for (size_t m = 0; m < organMeshes.size(); m++) {
                        best_vertices[m] = organMeshes[m]->mVertices;
                        best_normals[m] = organMeshes[m]->mNormals;
                    }
                    std::cout << "  New best fitness: " << best_fitness << std::endl;
                }

                prev_fitness = current_fitness;

            } else {
                for (size_t m = 0; m < organMeshes.size(); m++) {
                    organMeshes[m]->mVertices = backup_vertices[m];
                    organMeshes[m]->mNormals = backup_normals[m];
                    setUp(*organMeshes[m]);
                }
            }

            float fitness_change = std::abs(result.fitness - prev_fitness);

            std::cout << "\n  Summary:" << std::endl;
            std::cout << "  - Fitness: " << result.fitness;
            if (should_apply_transform) {
                std::cout << " (change: " << fitness_change << ")" << std::endl;
            } else {
                std::cout << " [SKIPPED]" << std::endl;
            }
            std::cout << "  - Best fitness so far: " << best_fitness << " (iter " << best_iteration << ")" << std::endl;
            std::cout << "  - Skipped count: " << skipped_count << "/" << (iter + 1) << std::endl;

            if (iter >= 3 && should_apply_transform) {
                if (result.fitness >= min_fitness_for_convergence && fitness_change < convergence_threshold) {
                    std::cout << "\n Converged!" << std::endl;
                    break;
                }
                if (result.fitness > 0.50f) {
                    std::cout << "\n Excellent fitness achieved!" << std::endl;
                    break;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "\n Exception in iteration " << (iter+1) << ": " << e.what() << std::endl;

            for (size_t m = 0; m < organMeshes.size(); m++) {
                organMeshes[m]->mVertices = best_vertices[m];
                organMeshes[m]->mNormals = best_normals[m];
                setUp(*organMeshes[m]);
            }

            if (iter > 0) {
                std::cout << " Rolled back to best state" << std::endl;
                break;
            } else {
                return false;
            }
        }
    }

    std::cout << "\n" << std::endl;
    if (best_iteration == 0) {
        std::cout << "  Keeping initial state (fitness=" << best_fitness << ")" << std::endl;
    } else {
        std::cout << "  Applying best result (iter " << best_iteration << ", fitness " << best_fitness << ")" << std::endl;
    }
    for (size_t m = 0; m < organMeshes.size(); m++) {
        organMeshes[m]->mVertices = best_vertices[m];
        organMeshes[m]->mNormals = best_normals[m];
        setUp(*organMeshes[m]);
    }

    std::cout << "\nStep 7: Extracting correspondences..." << std::endl;
    try {
        auto targetCloud = reg.extractFrontFacePoints(*screenMesh, gridWidth, gridHeight, zThreshold);
        auto sourceCloud = std::make_shared<PointCloud>();
        for (size_t idx : sourceVertexIndices) {
            if (idx * 3 + 2 < sourceMesh->mVertices.size()) {
                sourceCloud->addPoint(glm::vec3(
                    sourceMesh->mVertices[idx * 3],
                    sourceMesh->mVertices[idx * 3 + 1],
                    sourceMesh->mVertices[idx * 3 + 2]));
            }
        }

        std::vector<glm::vec3> source_points, target_points;
        reg.extractCorrespondences(sourceCloud, targetCloud, glm::mat4(1.0f), 1.0f,
                                   source_points, target_points);

        std::cout << "  Found " << source_points.size() << " correspondences" << std::endl;

        int max_markers = Reg3D::g_maxCorrespondenceMarkers;
        std::vector<glm::vec3> sampled_source, sampled_target;

        if (source_points.size() > static_cast<size_t>(max_markers)) {
            float step = static_cast<float>(source_points.size()) / max_markers;
            for (int i = 0; i < max_markers; i++) {
                int idx = static_cast<int>(i * step);
                if (idx < static_cast<int>(source_points.size())) {
                    sampled_source.push_back(source_points[idx]);
                    sampled_target.push_back(target_points[idx]);
                }
            }
        } else {
            sampled_source = source_points;
            sampled_target = target_points;
        }

        registrationHandle.objectPoints.clear();
        registrationHandle.boardPoints.clear();
        for (const auto& sp : sampled_source)
            registrationHandle.objectPoints.push_back(sp);
        for (const auto& tp : sampled_target)
            registrationHandle.boardPoints.push_back(tp);

        {
            float totalErr = 0.0f;
            float sumSq = 0.0f;
            float maxErr = 0.0f;
            for (size_t i = 0; i < sampled_source.size(); i++) {
                float d = glm::distance(sampled_source[i], sampled_target[i]);
                totalErr += d;
                sumSq += d * d;
                if (d > maxErr) maxErr = d;
            }
            float n = sampled_source.empty() ? 1.0f : (float)sampled_source.size();
            registrationHandle.averageError = totalErr / n;
            registrationHandle.rmse = std::sqrt(sumSq / n);
            registrationHandle.maxError = maxErr;
        }
        registrationHandle.scaleFactor = Reg3D::estimateScaleOnly(
            registrationHandle.objectPoints, registrationHandle.boardPoints);

        registrationHandle.fitness = best_fitness;
        registrationHandle.icpRmse = final_result.inlier_rmse;
        registrationHandle.bestIteration = best_iteration;
        registrationHandle.refineCount = 0;

        registrationHandle.useRegistration = true;
        registrationHandle.state = RegistrationData::REGISTERED;

    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not extract correspondences: " << e.what() << std::endl;
    }

    if (g_progressCallback) g_progressCallback(1.0f, "Hemi Auto: Complete!");

    std::cout << "\n================================================" << std::endl;
    std::cout << "| [OK] Registration Complete                 |" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "  Best fitness:       " << best_fitness << " (iteration " << best_iteration << ")" << std::endl;
    std::cout << "  Final RMSE:         " << final_result.inlier_rmse << std::endl;
    std::cout << "  Avg Error:          " << registrationHandle.averageError << std::endl;
    std::cout << "  RMSE:               " << registrationHandle.rmse << std::endl;
    std::cout << "  Max Error:          " << registrationHandle.maxError << std::endl;
    std::cout << "  Skipped iterations: " << skipped_count << std::endl;
    std::cout << "  Meshes transformed: " << organMeshes.size() << std::endl;

    return true;
}

}
