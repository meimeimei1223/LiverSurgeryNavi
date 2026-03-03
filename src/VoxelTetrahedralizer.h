#ifndef VOXEL_TETRAHEDRALIZER_H
#define VOXEL_TETRAHEDRALIZER_H

#include <vector>
#include <string>
#include <set>
#include <map>
#include <unordered_map>
#include <chrono>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include "MeshDataTypes.h"

class VoxelTetrahedralizer {
public:
    // MeshData構造体
    struct MeshData {
        std::vector<float> verts;
        std::vector<int> tetIds;
        std::vector<int> tetEdgeIds;
        std::vector<int> tetSurfaceTriIds;
        
        // スムージング後のデータ
        std::vector<float> smoothedVerts;
        std::vector<int> smoothedSurfaceTriIds;
        bool isSmoothed = false;
    };

    // ボクセル構造体
    struct Voxel {
        glm::vec3 min;
        glm::vec3 max;
        glm::vec3 center;
        int targetTriangleCount;
        bool isRemoved;
    };

    // バウンディングボックス構造体
    struct BoundingBox {
        glm::vec3 min;
        glm::vec3 max;
        glm::vec3 size;
        glm::vec3 center;
    };

    // デバッグ構造体
    struct DebugInfo {
        int totalTargetVertices = 0;
        int verticesInsideTets = 0;
        int verticesOutsideTets = 0;
        std::vector<int> outsideVertexIndices;
        std::map<int, std::vector<int>> vertexToTetMap;
        float coveragePercentage = 0.0f;
    };

    // 膨張設定構造体
    struct InflationSettings {
        bool enabled = true;
        float targetCoverage = 90.0f;
        float successThreshold = 99.0f;
        float baseInflateAmount = 0.01f;
        float lowCoverageInflate = 0.05f;
        float veryLowCoverageInflate = 0.03f;
        float inflateIncreaseRate = 1.5f;
        float inflateDecreaseRate = 0.7f;
        float fineAdjustThreshold = 98.0f;
        float fineAdjustAmount = 0.005f;
        int maxAttempts = 100;
        float slowProgressThreshold = 0.1f;
        float fastProgressThreshold = 5.0f;
    };

    // コンストラクタ
    VoxelTetrahedralizer(int gridSize, const std::string& targetPath, const std::string& outputTetPath);

    // データゲッター
    const MeshData& getVoxTetData() const;

    // スムージング設定
    void setSmoothingEnabled(bool enable, int iterations = 5, float factor = 0.5f, 
                            bool adjustSize = true, int scalingMethod = 2);

    // 膨張設定
    void setInflationSettings(const InflationSettings& settings);
    void setInflationEnabled(bool enabled);
    void setTargetCoverage(float coverage);
    void setInflationParameters(float baseAmount, float increaseRate = 1.5f, float decreaseRate = 0.7f);

    // メイン実行関数
    MeshDataTypes::SimpleMeshData execute();

    // デバッグ関数
    DebugInfo debugCheckTargetVertexCoverage();

    // メッシュ操作
    void inflateMesh(float inflateAmount);

    // エクスポート関数
    bool exportSmoothSurfaceToOBJ(const std::string& filename) const;
    bool exportCurrentMeshToFile(const std::string& filename, bool exportSmoothedVersion = true);
    bool exportCurrentSurfaceToOBJ(const std::string& filename);

    // 表示用メッシュデータ生成
    MeshDataTypes::SimpleMeshData generateDisplayMeshData();

private:
    // メンバ変数
    MeshData voxTetData;
    int gridSize_;
    std::string targetPath_;
    std::string outputTetPath_;
    
    bool enableSmoothing_ = false;
    int smoothingIterations_ = 5;
    float smoothingFactor_ = 0.5f;
    bool enableSizeAdjustment_ = true;
    int scalingMethod_ = 2;
    
    InflationSettings inflationSettings_;
    
    std::vector<GLfloat> cubeVertices_;
    std::vector<GLuint> cubeIndices_;
    std::vector<GLfloat> targetVertices_;
    std::vector<GLuint> targetIndices_;
    
    std::vector<std::vector<std::vector<Voxel>>> voxels_;
    glm::vec3 gridMin_, gridMax_;
    glm::vec3 voxelSize_;
    
    std::chrono::high_resolution_clock::time_point totalStartTime_;
    std::chrono::milliseconds step1Duration_, step2Duration_, step3Duration_, step4Duration_;

    // プライベートメソッド（宣言のみ）
    void performAutomaticInflation();
    bool isPointInTetrahedron(const glm::vec3& point, size_t tetIdx);
    bool isPointInTetrahedronBarycentric(const glm::vec3& p, const glm::vec3& a, 
                                        const glm::vec3& b, const glm::vec3& c, const glm::vec3& d);
    void adjustInternalVertices(const std::set<int>& surfaceVertices, float adjustAmount);
    void initializeCube();
    BoundingBox getBoundingBoxFromVertices(const std::vector<GLfloat>& vertices);
    BoundingBox calculateBoundingBox(const std::vector<float>& vertices, const std::set<int>& surfaceVertices);
    bool loadTargetMesh();
    void prepareCube();
    void initializeVoxelGrid();
    bool isPointInsideTarget(const glm::vec3& point);
    void checkVoxelCenters();
    void carveExternalVoxels();
    void generateTetrahedra();
    void applySmoothingToMeshData();
    void generateSmoothSurface();
    void applyLaplacianSmoothing();
    void adjustMeshSize(const std::set<int>& surfaceVertices);
    void writeOutputFile(const std::vector<glm::vec3>& vertices,
                        const std::vector<std::vector<int>>& tetrahedra,
                        const std::vector<std::vector<int>>& surfaceTriangles);
public:
    // メンバ変数に追加
    bool verbose_ = false;

    // publicメソッドに追加
    void setVerbose(bool v);
};

#endif // VOXEL_TETRAHEDRALIZER_H
