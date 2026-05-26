#ifndef TETOMESHDATA_H
#define TETOMESHDATA_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <cmath>
#include <atomic>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp> // glm::inverse
#include "SoftBody.h"

class TetoMeshData {
public:
    // OBJファイルから読み込み、VetexAndFaceのみ読み込みMeshDataに変換する
    static SoftBody::MeshData ReadVetexAndFace(const std::string& objPath) {
        std::cout << "Starting tetrahedralization..." << std::endl;
        SoftBody::MeshData meshData;
        std::vector<glm::vec3> vertices;
        std::vector<std::vector<int>> faces;

        if (!loadOBJ(objPath, vertices, faces)) {
            std::cerr << "Failed to load OBJ file: " << objPath << std::endl;
            return meshData;
        }
        // MeshData用に平坦化した頂点リストを作成
        meshData.verts.clear();
        for (const auto& v : vertices) {
            meshData.verts.push_back(v.x);
            meshData.verts.push_back(v.y);
            meshData.verts.push_back(v.z);
        }
        // OBJの面情報（表面三角形）はそのまま保持（※テトラ分割と完全に一致しない場合もあります）
        meshData.tetSurfaceTriIds.clear();
        for (const auto& face : faces) {
            for (int idx : face) {
                meshData.tetSurfaceTriIds.push_back(idx);
            }
        }
        printMeshStatistics(meshData);

        return meshData;
    }

    // Softbody Vispositonを読み込みMeshDataに変換する
    static SoftBody::MeshData ReadVisPosAndFace(const std::vector<float>& vis_position, const std::vector<int>& vis_TriIds) {

        SoftBody::MeshData meshData;
        std::vector<glm::vec3> vertices;
        std::vector<std::vector<int>> faces;

        // 平坦なvis_position配列をglm::vec3頂点に変換
        for (size_t i = 0; i < vis_position.size(); i += 3) {
            if (i + 2 < vis_position.size()) {
                vertices.push_back(glm::vec3(
                    vis_position[i],
                    vis_position[i + 1],
                    vis_position[i + 2]
                ));
            }
        }

        // 平坦な三角形インデックス配列をfaces（面）に変換
        for (size_t i = 0; i < vis_TriIds.size(); i += 3) {
            if (i + 2 < vis_TriIds.size()) {
                std::vector<int> face = {
                    vis_TriIds[i],
                    vis_TriIds[i + 1],
                    vis_TriIds[i + 2]
                };
                faces.push_back(face);
            }
        }

        // MeshData用に平坦化した頂点リストを作成
        meshData.verts.clear();
        for (const auto& v : vertices) {
            meshData.verts.push_back(v.x);
            meshData.verts.push_back(v.y);
            meshData.verts.push_back(v.z);
        }

        // 三角形面情報を設定
        meshData.tetSurfaceTriIds.clear();
        for (const auto& face : faces) {
            for (int idx : face) {
                meshData.tetSurfaceTriIds.push_back(idx);
            }
        }

        // 注意: これは表示用メッシュのみを作成するため、
        // 実際には四面体化（tetrahedralization）を行っていません。
        // tetIdsとtetEdgeIdsは空のままです。
        meshData.tetIds.clear();
        meshData.tetEdgeIds.clear();

        printMeshStatistics(meshData);

        return meshData;
    }


protected:  // privateからprotectedに変更
    static bool loadOBJ(const std::string& filename,
                       std::vector<glm::vec3>& vertices,
                       std::vector<std::vector<int>>& faces){
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            if (type == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                vertices.emplace_back(x, y, z);
            } else if (type == "f") {
                std::vector<int> face;
                std::string vertex;
                while (iss >> vertex) {
                    size_t pos = vertex.find('/');
                    if (pos != std::string::npos)
                        vertex = vertex.substr(0, pos);
                    face.push_back(std::stoi(vertex) - 1);
                }
                if (face.size() == 3)
                    faces.push_back(face);
            }
        }
        return !vertices.empty();
    }

private:
    // テトラヘドロン情報：各テトラは4頂点インデックスと外接球の中心・半径²を保持
    struct Tetrahedron {
        std::array<int, 4> vertices;
        glm::vec3 circumCenter;
        float circumRadiusSquared;
        Tetrahedron(int v0, int v1, int v2, int v3)
            : vertices({v0, v1, v2, v3}), circumCenter(0.0f), circumRadiusSquared(0.0f) {}
    };

    // 統計情報の出力
    static void printMeshStatistics(const SoftBody::MeshData& meshData) {
        std::cout << "Mesh generation completed:" << std::endl;
        std::cout << "Vertices: " << meshData.verts.size() / 3 << std::endl;
        std::cout << "Tetrahedra: " << meshData.tetIds.size() / 4 << std::endl;
        std::cout << "Surface triangles: " << meshData.tetSurfaceTriIds.size() / 3 << std::endl;
        std::cout << "Edges: " << meshData.tetEdgeIds.size() / 2 << std::endl;
    }
};

#endif // TETOMESHDATA_H
