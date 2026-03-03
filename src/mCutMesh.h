#ifndef MCUTMESH_H
#define MCUTMESH_H
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "ShaderProgram.h"
#include "SimpleCamera.hpp"
#include "PlatformCompat.h"

#ifndef STBI_INCLUDE_STB_IMAGE_H
extern unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
extern void stbi_image_free(void *retval_from_stbi_load);
#endif
class ShaderProgram;
struct mCutMesh {
    GLuint VAO = 0, VBO = 0, EBO = 0, NBO = 0, TBO = 0, textureID = 0;
    std::vector<GLfloat> mVertices;
    std::vector<GLfloat> mNormals;
    std::vector<GLfloat> mTexCoords;
    std::vector<GLuint> mIndices;
    int numFaces;
    glm::vec3 mColor;
    bool hasTexture = false;
    std::vector<unsigned char> loadedImageData;
    int loadedImageWidth = 0;
    int loadedImageHeight = 0;
    int loadedImageChannels = 0;
    std::vector<unsigned char> depthImageData;
    int depthWidth = 0;
    int depthHeight = 0;
    SimpleCamera* camera = nullptr;
    std::vector<unsigned char> cameraFrame;
    bool useLiveCamera = false;
    bool exportObjFile(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not create file " << filepath << std::endl;
            return false;
        }
        file << "# OBJ File exported from mCutMesh" << std::endl;
        file << "# Vertices: " << (mVertices.size() / 3) << std::endl;
        file << "# Faces: " << (mIndices.size() / 3) << std::endl;
        file << std::endl;
        for (size_t i = 0; i < mVertices.size(); i += 3) {
            file << "v " << mVertices[i] << " "
                 << mVertices[i + 1] << " "
                 << mVertices[i + 2] << std::endl;
        }
        file << std::endl;
        if (!mTexCoords.empty()) {
            for (size_t i = 0; i < mTexCoords.size(); i += 2) {
                file << "vt " << mTexCoords[i] << " "
                     << mTexCoords[i + 1] << std::endl;
            }
            file << std::endl;
        }
        if (!mNormals.empty()) {
            for (size_t i = 0; i < mNormals.size(); i += 3) {
                file << "vn " << mNormals[i] << " "
                     << mNormals[i + 1] << " "
                     << mNormals[i + 2] << std::endl;
            }
            file << std::endl;
        }
        bool hasNormals = !mNormals.empty();
        bool hasTexCoords = !mTexCoords.empty();
        for (size_t i = 0; i < mIndices.size(); i += 3) {
            file << "f ";
            for (int j = 0; j < 3; j++) {
                GLuint idx = mIndices[i + j] + 1;
                if (hasTexCoords && hasNormals) {
                    file << idx << "/" << idx << "/" << idx;
                } else if (hasTexCoords) {
                    file << idx << "/" << idx;
                } else if (hasNormals) {
                    file << idx << "//" << idx;
                } else {
                    file << idx;
                }
                if (j < 2) file << " ";
            }
            file << std::endl;
        }
        file.close();
        std::cout << "Successfully exported OBJ file: " << filepath << std::endl;
        std::cout << "  Vertices: " << (mVertices.size() / 3) << std::endl;
        std::cout << "  Faces: " << (mIndices.size() / 3) << std::endl;
        return true;
    }
    void loadTextureFromData(const unsigned char* data, int width, int height, int channels) {
        if (!data || width <= 0 || height <= 0) {
            std::cerr << "Error: Invalid image data provided" << std::endl;
            return;
        }
        size_t dataSize = static_cast<size_t>(width) * height * channels;
        loadedImageData.assign(data, data + dataSize);
        loadedImageWidth = width;
        loadedImageHeight = height;
        loadedImageChannels = channels;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        std::vector<unsigned char> flipped(dataSize);
        int rowSize = width * channels;
        for (int y = 0; y < height; y++) {
            memcpy(&flipped[y * rowSize], &data[(height - 1 - y) * rowSize], rowSize);
        }
        GLenum format = GL_RGB;
        if (channels == 4) {
            format = GL_RGBA;
        } else if (channels == 1) {
            format = GL_RED;
        }

        // â˜… ã‚¢ãƒ©ã‚¤ãƒ¡ãƒ³ãƒˆä¿®æ­£ï¼ˆè¡Œãƒã‚¤ãƒˆæ•°ãŒ4ã®å€æ•°ã§ãªã„ç”»åƒã§ã‚¯ãƒ©ãƒƒã‚·ãƒ¥é˜²æ­¢ï¼‰
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, flipped.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        hasTexture = true;
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    void loadTextureFromFile(const std::string& filepath) {
        int width, height, channels;
        unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
        if (data) {
            loadTextureFromData(data, width, height, channels);
            stbi_image_free(data);
        } else {
            std::cerr << "Error: Could not load texture from " << filepath << std::endl;
        }
    }
    bool initCamera(int deviceIndex = 0, int width = 640, int height = 480) {
        if (camera) { camera->close(); delete camera; camera = nullptr; }
        camera = new SimpleCamera();
        auto devices = SimpleCamera::listDevices();
        std::cout << "=== Camera devices ===" << std::endl;
        for (size_t i = 0; i < devices.size(); i++)
            std::cout << "[" << i << "] " << devices[i] << std::endl;
        if (devices.empty()) {
            std::cerr << "No camera devices found" << std::endl;
            delete camera; camera = nullptr; return false;
        }
#ifndef STB_IMAGE_IMPLEMENTATION
        std::cerr << "WARNING: STB_IMAGE_IMPLEMENTATION not defined!" << std::endl;
        std::cerr << "  MJPEG camera decode will NOT work." << std::endl;
        std::cerr << "  Add this to main.cpp BEFORE #include mCutMesh.h:" << std::endl;
        std::cerr << "    #define STB_IMAGE_IMPLEMENTATION" << std::endl;
        std::cerr << "    #include \"stb_image.h\"" << std::endl;
#endif
        if (!camera->open(deviceIndex, width, height)) {
            std::cerr << "Camera open failed (device " << deviceIndex << ")" << std::endl;
            delete camera; camera = nullptr; return false;
        }
        int w = camera->getWidth(), h = camera->getHeight();
        loadedImageWidth = w;
        loadedImageHeight = h;
        loadedImageChannels = 3;
        size_t sz = static_cast<size_t>(w) * h * 3;
        loadedImageData.assign(sz, 128);
        cameraFrame.resize(sz);
        std::cout << "Camera warming up..." << std::endl;
        bool gotValidFrame = false;
        for (int i = 0; i < 30; i++) {
            if (camera->captureFrame(cameraFrame)) {
                bool allZero = true;
                for (size_t j = 0; j < std::min(sz, (size_t)1000); j++) {
                    if (cameraFrame[j] != 0) { allZero = false; break; }
                }
                if (allZero) {
                    std::cerr << "  Frame " << i << ": captured but data is all zeros (MJPEG decode failed?)" << std::endl;
                } else {
                    std::cout << "  Camera ready (frame " << i << ", first bytes: "
                              << (int)cameraFrame[0] << " " << (int)cameraFrame[1] << " " << (int)cameraFrame[2] << ")" << std::endl;
                    loadedImageData = cameraFrame;
                    gotValidFrame = true;
                    break;
                }
            }
            usleep(50000);
        }
        if (!gotValidFrame) {
            std::cerr << "WARNING: Could not get valid frame from camera." << std::endl;
            std::cerr << "  If MJPEG format, ensure STB_IMAGE_IMPLEMENTATION is defined in main.cpp" << std::endl;
        }
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, loadedImageData.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        std::cout << "  textureID=" << textureID << " hasTexture=true" << std::endl;
        hasTexture = true;
        useLiveCamera = true;
        std::cout << "Camera initialized: " << w << "x" << h << std::endl;
        return true;
    }
    int cameraUpdateCount = 0;
    std::vector<unsigned char> cameraFlipBuffer;
    bool updateTextureFromCamera() {
        if (!camera || !camera->isOpened() || !useLiveCamera) return false;
        if (camera->captureFrame(cameraFrame)) {
            int w = camera->getWidth(), h = camera->getHeight();
            cameraFlipBuffer.resize(cameraFrame.size());
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int srcIdx = ((h - 1 - y) * w + (w - 1 - x)) * 3;
                    int dstIdx = (y * w + x) * 3;
                    cameraFlipBuffer[dstIdx + 0] = cameraFrame[srcIdx + 0];
                    cameraFlipBuffer[dstIdx + 1] = cameraFrame[srcIdx + 1];
                    cameraFlipBuffer[dstIdx + 2] = cameraFrame[srcIdx + 2];
                }
            }
            if (cameraUpdateCount < 3) {
                bool allZero = true;
                for (size_t j = 0; j < std::min(cameraFlipBuffer.size(), (size_t)100); j++) {
                    if (cameraFlipBuffer[j] != 0) { allZero = false; break; }
                }
                std::cout << "Camera frame " << cameraUpdateCount
                          << ": " << (allZero ? "ALL ZEROS (decode failed)" : "OK")
                          << " textureID=" << textureID << std::endl;
                cameraUpdateCount++;
            }
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                            GL_RGB, GL_UNSIGNED_BYTE, cameraFlipBuffer.data());
            glBindTexture(GL_TEXTURE_2D, 0);
            return true;
        }
        return false;
    }
    void closeCamera() {
        if (camera) { camera->close(); delete camera; camera = nullptr; }
        useLiveCamera = false;
    }
    bool isUsingCamera() const {
        return useLiveCamera && camera && camera->isOpened();
    }
    void generatePlateFromImage(float thickness = 0.1f) {
        if (loadedImageData.empty()) {
            std::cerr << "Error: No image loaded" << std::endl;
            return;
        }
        float aspectRatio = static_cast<float>(loadedImageWidth) / static_cast<float>(loadedImageHeight);
        float width = aspectRatio;
        float height = 1.0f;
        float halfThickness = thickness / 2.0f;
        mVertices.clear();
        mNormals.clear();
        mTexCoords.clear();
        mIndices.clear();
        std::vector<glm::vec3> vertices = {
            glm::vec3(-width/2, -height/2,  halfThickness),
            glm::vec3( width/2, -height/2,  halfThickness),
            glm::vec3( width/2,  height/2,  halfThickness),
            glm::vec3(-width/2,  height/2,  halfThickness),
            glm::vec3(-width/2, -height/2, -halfThickness),
            glm::vec3( width/2, -height/2, -halfThickness),
            glm::vec3( width/2,  height/2, -halfThickness),
            glm::vec3(-width/2,  height/2, -halfThickness)
        };
        std::vector<glm::vec2> frontTexCoords = {
            glm::vec2(0.0f, 0.0f),
            glm::vec2(1.0f, 0.0f),
            glm::vec2(1.0f, 1.0f),
            glm::vec2(0.0f, 1.0f)
        };
        std::vector<glm::vec3> faceNormals = {
            glm::vec3( 0.0f,  0.0f,  1.0f),
            glm::vec3( 0.0f,  0.0f, -1.0f),
            glm::vec3( 0.0f,  1.0f,  0.0f),
            glm::vec3( 0.0f, -1.0f,  0.0f),
            glm::vec3( 1.0f,  0.0f,  0.0f),
            glm::vec3(-1.0f,  0.0f,  0.0f)
        };
        int faceIndices[6][4] = {
            {0, 1, 2, 3},
            {5, 4, 7, 6},
            {3, 2, 6, 7},
            {4, 5, 1, 0},
            {1, 5, 6, 2},
            {4, 0, 3, 7}
        };
        for (int face = 0; face < 6; face++) {
            for (int i = 0; i < 4; i++) {
                int vertIdx = faceIndices[face][i];
                mVertices.push_back(vertices[vertIdx].x);
                mVertices.push_back(vertices[vertIdx].y);
                mVertices.push_back(vertices[vertIdx].z);
                mNormals.push_back(faceNormals[face].x);
                mNormals.push_back(faceNormals[face].y);
                mNormals.push_back(faceNormals[face].z);
                if (face == 0) {
                    mTexCoords.push_back(frontTexCoords[i].x);
                    mTexCoords.push_back(frontTexCoords[i].y);
                } else {
                    mTexCoords.push_back(-1.0f);
                    mTexCoords.push_back(-1.0f);
                }
            }
            int baseIdx = face * 4;
            mIndices.push_back(baseIdx + 0);
            mIndices.push_back(baseIdx + 1);
            mIndices.push_back(baseIdx + 2);
            mIndices.push_back(baseIdx + 0);
            mIndices.push_back(baseIdx + 2);
            mIndices.push_back(baseIdx + 3);
        }
        numFaces = 12;
    }
    void generateCubeTexCoords() {
        mTexCoords.clear();
        if (mVertices.size() == 24 * 3) {
            std::vector<glm::vec2> faceUVs = {
                {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}
            };
            for (int face = 0; face < 6; face++) {
                for (const auto& uv : faceUVs) {
                    mTexCoords.push_back(uv.x);
                    mTexCoords.push_back(uv.y);
                }
            }
        } else {
            for (size_t i = 0; i < mVertices.size(); i += 3) {
                float x = mVertices[i];
                float y = mVertices[i + 1];
                float z = mVertices[i + 2];
                float u = 0.5f + atan2(z, x) / (2.0f * M_PI);
                float v = 0.5f + asin(y) / M_PI;
                mTexCoords.push_back(u);
                mTexCoords.push_back(v);
            }
        }
    }
    void cleanup() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &NBO);
        glDeleteBuffers(1, &EBO);
        glDeleteBuffers(1, &TBO);
        if (hasTexture) {
            glDeleteTextures(1, &textureID);
        }
    }
    mCutMesh loadMeshFromFile(const char* filePath) {
        mCutMesh mesh;
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filePath << std::endl;
            return mesh;
        }
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec2> texCoords;
        std::vector<std::vector<int>> faces;
        std::vector<std::vector<int>> texIndices;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            if (type == "v") {
                float x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(glm::vec3(x, y, z));
            }
            else if (type == "vt") {
                float u, v;
                iss >> u >> v;
                texCoords.push_back(glm::vec2(u, v));
            }
            else if (type == "f") {
                std::vector<int> face;
                std::vector<int> texFace;
                std::string vertex;
                while (iss >> vertex) {
                    size_t pos1 = vertex.find('/');
                    size_t pos2 = vertex.find('/', pos1 + 1);
                    std::string vIdx = vertex.substr(0, pos1);
                    face.push_back(std::stoi(vIdx) - 1);
                    if (pos1 != std::string::npos && pos2 != std::string::npos) {
                        std::string vtIdx = vertex.substr(pos1 + 1, pos2 - pos1 - 1);
                        if (!vtIdx.empty()) {
                            texFace.push_back(std::stoi(vtIdx) - 1);
                        }
                    }
                }
                if (face.size() >= 3) {
                    for (size_t i = 1; i < face.size() - 1; ++i) {
                        std::vector<int> triangle = {face[0], face[i], face[i + 1]};
                        faces.push_back(triangle);
                        if (!texFace.empty() && texFace.size() == face.size()) {
                            std::vector<int> texTriangle = {texFace[0], texFace[i], texFace[i + 1]};
                            texIndices.push_back(texTriangle);
                        }
                    }
                }
            }
        }
        file.close();
        mesh.mVertices.clear();
        mesh.mVertices.reserve(vertices.size() * 3);
        for (const auto& vertex : vertices) {
            mesh.mVertices.push_back(vertex.x);
            mesh.mVertices.push_back(vertex.y);
            mesh.mVertices.push_back(vertex.z);
        }
        mesh.mIndices.clear();
        mesh.mIndices.reserve(faces.size() * 3);
        for (const auto& face : faces) {
            for (int idx : face) {
                mesh.mIndices.push_back(static_cast<GLuint>(idx));
            }
        }
        if (texCoords.empty() || texIndices.empty()) {
            mesh.generateCubeTexCoords();
        } else {
            mesh.mTexCoords.clear();
            mesh.mTexCoords.reserve(vertices.size() * 2);
            for (size_t i = 0; i < vertices.size(); i++) {
                if (i < texCoords.size()) {
                    mesh.mTexCoords.push_back(texCoords[i].x);
                    mesh.mTexCoords.push_back(texCoords[i].y);
                } else {
                    mesh.mTexCoords.push_back(0.0f);
                    mesh.mTexCoords.push_back(0.0f);
                }
            }
        }
        mesh.numFaces = faces.size();
        mesh.mColor = glm::vec3(0.7f, 0.7f, 0.7f);
        return mesh;
    }
    void generateGridPlaneWithSides(int gridWidth, int gridHeight, float thickness = 0.05f) {
        mVertices.clear();
        mNormals.clear();
        mTexCoords.clear();
        mIndices.clear();
        float planeWidth = (float)gridWidth / gridHeight;
        float planeHeight = 1.0f;
        float halfThickness = thickness / 2.0f;
        std::cout << "Generating grid plane with sides: " << (gridWidth+1) << "x" << (gridHeight+1)
                  << " vertices on front" << std::endl;
        for (int y = 0; y <= gridHeight; y++) {
            for (int x = 0; x <= gridWidth; x++) {
                float u = (float)x / gridWidth;
                float v = (float)y / gridHeight;
                float posX = (u - 0.5f) * planeWidth;
                float posY = (0.5f - v) * planeHeight;
                float posZ = halfThickness;
                mVertices.push_back(posX);
                mVertices.push_back(posY);
                mVertices.push_back(posZ);
                mNormals.push_back(0.0f);
                mNormals.push_back(0.0f);
                mNormals.push_back(1.0f);
                mTexCoords.push_back(u);
                mTexCoords.push_back(1.0f - v);
            }
        }
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                GLuint topLeft = y * (gridWidth + 1) + x;
                GLuint topRight = topLeft + 1;
                GLuint bottomLeft = (y + 1) * (gridWidth + 1) + x;
                GLuint bottomRight = bottomLeft + 1;
                mIndices.push_back(topLeft);
                mIndices.push_back(bottomLeft);
                mIndices.push_back(topRight);
                mIndices.push_back(topRight);
                mIndices.push_back(bottomLeft);
                mIndices.push_back(bottomRight);
            }
        }
        GLuint backStart = mVertices.size() / 3;
        float backZ = -halfThickness;
        mVertices.insert(mVertices.end(), {
                                              -planeWidth/2, -planeHeight/2, backZ,
                                              planeWidth/2, -planeHeight/2, backZ,
                                              planeWidth/2,  planeHeight/2, backZ,
                                              -planeWidth/2,  planeHeight/2, backZ
                                          });
        for(int i=0; i<4; i++) {
            mNormals.insert(mNormals.end(), {0.0f, 0.0f, -1.0f});
            mTexCoords.insert(mTexCoords.end(), {-1.0f, -1.0f});
        }
        mIndices.push_back(backStart + 1);
        mIndices.push_back(backStart + 0);
        mIndices.push_back(backStart + 2);
        mIndices.push_back(backStart + 2);
        mIndices.push_back(backStart + 0);
        mIndices.push_back(backStart + 3);
        GLuint frontBL = 0;
        GLuint frontBR = gridWidth;
        GLuint frontTR = gridHeight * (gridWidth + 1) + gridWidth;
        GLuint frontTL = gridHeight * (gridWidth + 1);
        GLuint sideStart = mVertices.size() / 3;
        mVertices.insert(mVertices.end(), {
                                              mVertices[frontBL*3], mVertices[frontBL*3+1], mVertices[frontBL*3+2],
                                              mVertices[frontBR*3], mVertices[frontBR*3+1], mVertices[frontBR*3+2],
                                              planeWidth/2, -planeHeight/2, backZ,
                                              -planeWidth/2, -planeHeight/2, backZ
                                          });
        for(int i=0; i<4; i++) {
            mNormals.insert(mNormals.end(), {0.0f, -1.0f, 0.0f});
            mTexCoords.insert(mTexCoords.end(), {-1.0f, -1.0f});
        }
        mIndices.insert(mIndices.end(), {sideStart+0, sideStart+2, sideStart+1, sideStart+0, sideStart+3, sideStart+2});
        sideStart = mVertices.size() / 3;
        mVertices.insert(mVertices.end(), {
                                              mVertices[frontTL*3], mVertices[frontTL*3+1], mVertices[frontTL*3+2],
                                              mVertices[frontTR*3], mVertices[frontTR*3+1], mVertices[frontTR*3+2],
                                              planeWidth/2, planeHeight/2, backZ,
                                              -planeWidth/2, planeHeight/2, backZ
                                          });
        for(int i=0; i<4; i++) {
            mNormals.insert(mNormals.end(), {0.0f, 1.0f, 0.0f});
            mTexCoords.insert(mTexCoords.end(), {-1.0f, -1.0f});
        }
        mIndices.insert(mIndices.end(), {sideStart+0, sideStart+1, sideStart+2, sideStart+0, sideStart+2, sideStart+3});
        sideStart = mVertices.size() / 3;
        mVertices.insert(mVertices.end(), {
                                              mVertices[frontBR*3], mVertices[frontBR*3+1], mVertices[frontBR*3+2],
                                              mVertices[frontTR*3], mVertices[frontTR*3+1], mVertices[frontTR*3+2],
                                              planeWidth/2, planeHeight/2, backZ,
                                              planeWidth/2, -planeHeight/2, backZ
                                          });
        for(int i=0; i<4; i++) {
            mNormals.insert(mNormals.end(), {1.0f, 0.0f, 0.0f});
            mTexCoords.insert(mTexCoords.end(), {-1.0f, -1.0f});
        }
        mIndices.insert(mIndices.end(), {sideStart+0, sideStart+1, sideStart+2, sideStart+0, sideStart+2, sideStart+3});
        sideStart = mVertices.size() / 3;
        mVertices.insert(mVertices.end(), {
                                              mVertices[frontBL*3], mVertices[frontBL*3+1], mVertices[frontBL*3+2],
                                              mVertices[frontTL*3], mVertices[frontTL*3+1], mVertices[frontTL*3+2],
                                              -planeWidth/2, planeHeight/2, backZ,
                                              -planeWidth/2, -planeHeight/2, backZ
                                          });
        for(int i=0; i<4; i++) {
            mNormals.insert(mNormals.end(), {-1.0f, 0.0f, 0.0f});
            mTexCoords.insert(mTexCoords.end(), {-1.0f, -1.0f});
        }
        mIndices.insert(mIndices.end(), {sideStart+0, sideStart+3, sideStart+1, sideStart+1, sideStart+3, sideStart+2});
        numFaces = mIndices.size() / 3;
        std::cout << "Generated: " << (mVertices.size()/3) << " vertices, "
                  << numFaces << " triangles" << std::endl;
    }
    bool loadDepthImage(const std::string& filepath, int targetWidth, int targetHeight) {
        int rawW, rawH, rawCh;
        unsigned char* rawData = stbi_load(filepath.c_str(), &rawW, &rawH, &rawCh, 1);
        if (!rawData) {
            std::cerr << "Error: Could not load depth image from " << filepath << std::endl;
            return false;
        }
        std::cout << "Loaded depth image: " << rawW << "x" << rawH << std::endl;
        depthWidth = targetWidth;
        depthHeight = targetHeight;
        depthImageData.resize(static_cast<size_t>(targetWidth) * targetHeight);
        for (int ty = 0; ty < targetHeight; ty++) {
            for (int tx = 0; tx < targetWidth; tx++) {
                float srcX = static_cast<float>(tx) * rawW / targetWidth;
                float srcY = static_cast<float>(ty) * rawH / targetHeight;
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                int x1 = std::min(x0 + 1, rawW - 1);
                int y1 = std::min(y0 + 1, rawH - 1);
                float fx = srcX - x0;
                float fy = srcY - y0;
                float val = rawData[y0 * rawW + x0] * (1 - fx) * (1 - fy)
                            + rawData[y0 * rawW + x1] * fx * (1 - fy)
                            + rawData[y1 * rawW + x0] * (1 - fx) * fy
                            + rawData[y1 * rawW + x1] * fx * fy;
                depthImageData[ty * targetWidth + tx] = static_cast<unsigned char>(val + 0.5f);
            }
        }
        stbi_image_free(rawData);
        std::cout << "Resized to: " << depthWidth << "x" << depthHeight << std::endl;
        return true;
    }
    std::vector<float> applyGaussianSmoothingKernel(const std::vector<float>& depths,
                                                    int gridWidth, int gridHeight,
                                                    int iterations = 2) {
        std::vector<float> smoothed = depths;
        std::vector<float> temp(depths.size());
        float kernel[5][5] = {
            {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f},
            {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
            {0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f},
            {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
            {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f}
        };
        for (int iter = 0; iter < iterations; iter++) {
            for (int y = 0; y <= gridHeight; y++) {
                for (int x = 0; x <= gridWidth; x++) {
                    int idx = y * (gridWidth + 1) + x;
                    float sum = 0.0f;
                    float weightSum = 0.0f;
                    for (int ky = -2; ky <= 2; ky++) {
                        for (int kx = -2; kx <= 2; kx++) {
                            int ny = y + ky;
                            int nx = x + kx;
                            if (ny >= 0 && ny <= gridHeight &&
                                nx >= 0 && nx <= gridWidth) {
                                int nidx = ny * (gridWidth + 1) + nx;
                                float weight = kernel[ky + 2][kx + 2];
                                sum += smoothed[nidx] * weight;
                                weightSum += weight;
                            }
                        }
                    }
                    temp[idx] = sum / weightSum;
                }
            }
            smoothed = temp;
        }
        return smoothed;
    }
    std::vector<float> applyGaussianSmoothing(const std::vector<float>& depths,
                                              int gridWidth, int gridHeight,
                                              int iterations = 2) {
        std::vector<float> smoothed = depths;
        std::vector<float> temp(depths.size());
        float kernel[3][3] = {
            {0.0625f, 0.125f, 0.0625f},
            {0.125f,  0.25f,  0.125f},
            {0.0625f, 0.125f, 0.0625f}
        };
        for (int iter = 0; iter < iterations; iter++) {
            for (int y = 0; y <= gridHeight; y++) {
                for (int x = 0; x <= gridWidth; x++) {
                    int idx = y * (gridWidth + 1) + x;
                    float sum = 0.0f;
                    float weightSum = 0.0f;
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int ny = y + ky;
                            int nx = x + kx;
                            if (ny >= 0 && ny <= gridHeight &&
                                nx >= 0 && nx <= gridWidth) {
                                int nidx = ny * (gridWidth + 1) + nx;
                                float weight = kernel[ky + 1][kx + 1];
                                sum += smoothed[nidx] * weight;
                                weightSum += weight;
                            }
                        }
                    }
                    temp[idx] = sum / weightSum;
                }
            }
            smoothed = temp;
        }
        return smoothed;
    }
    std::vector<float> applyEdgeMask(const std::vector<float>& depths,
                                     int gridWidth, int gridHeight,
                                     float centerRatio = 0.9f) {
        std::vector<float> masked = depths;
        float centerWidth = gridWidth * centerRatio / 2.0f;
        float centerHeight = gridHeight * centerRatio / 2.0f;
        float centerX = gridWidth / 2.0f;
        float centerY = gridHeight / 2.0f;
        std::cout << "Masking edges: keeping center "
                  << (centerRatio * 100) << "% region" << std::endl;
        int maskedCount = 0;
        for (int y = 0; y <= gridHeight; y++) {
            for (int x = 0; x <= gridWidth; x++) {
                int idx = y * (gridWidth + 1) + x;
                float dx = std::abs(x - centerX);
                float dy = std::abs(y - centerY);
                if (dx > centerWidth || dy > centerHeight) {
                    masked[idx] = 0.0f;
                    maskedCount++;
                }
            }
        }
        std::cout << "Masked " << maskedCount << " edge vertices (set to flat)" << std::endl;
        return masked;
    }
    std::vector<float> calculateNormalizedDepth(int gridWidth, int gridHeight,
                                                float percentile = 0.95f,
                                                float maskRatio = 0.9f) {
        if (depthImageData.empty()) {
            std::cerr << "Error: No depth image loaded" << std::endl;
            return std::vector<float>();
        }
        int imgWidth = depthWidth;
        int imgHeight = depthHeight;
        float blockWidthF  = (float)imgWidth  / gridWidth;
        float blockHeightF = (float)imgHeight / gridHeight;
        int halfBlockW = std::max(1, (int)(blockWidthF  / 2.0f));
        int halfBlockH = std::max(1, (int)(blockHeightF / 2.0f));
        std::cout << "Block size: " << blockWidthF << "x" << blockHeightF
                  << " pixels (float)" << std::endl;
        std::vector<float> rawDepths;
        rawDepths.reserve((gridWidth + 1) * (gridHeight + 1));
        for (int gy = 0; gy <= gridHeight; gy++) {
            for (int gx = 0; gx <= gridWidth; gx++) {
                int centerX = (int)((float)gx / gridWidth  * (imgWidth  - 1));
                int centerY = (int)((float)gy / gridHeight * (imgHeight - 1));
                int startX = std::max(0, centerX - halfBlockW);
                int endX   = std::min(imgWidth,  centerX + halfBlockW + 1);
                int startY = std::max(0, centerY - halfBlockH);
                int endY   = std::min(imgHeight, centerY + halfBlockH + 1);
                float sum = 0.0f;
                int count = 0;
                for (int y = startY; y < endY; y++) {
                    for (int x = startX; x < endX; x++) {
                        float val = depthImageData[y * depthWidth + x];
                        if (val > 0.0f) {   // ãƒžã‚¹ã‚¯å¤–(æ·±åº¦0)ã‚’é™¤å¤–
                            sum += val;
                            count++;
                        }
                    }
                }
                float avgDepth = (count > 0) ? (sum / count) : 0.0f;
                rawDepths.push_back(avgDepth);
            }
        }
        std::vector<float> sortedDepths = rawDepths;
        std::sort(sortedDepths.begin(), sortedDepths.end());
        int percentileIndex = static_cast<int>(sortedDepths.size() * percentile);
        float maxDepth = sortedDepths[percentileIndex];
        // minDepth: 0ã‚’é™¤ã„ãŸæœ€å°å€¤ã‚’ä½¿ç”¨ï¼ˆãƒžã‚¹ã‚¯å¤–ã‚’ç„¡è¦–ï¼‰
        float minDepth = 0.0f;
        for (float d : sortedDepths) {
            if (d > 0.0f) { minDepth = d; break; }
        }
        std::cout << "Raw depth range: " << minDepth << " - " << maxDepth
                  << " (95th percentile, excluding mask)" << std::endl;
        std::vector<float> normalizedDepths;
        normalizedDepths.reserve(rawDepths.size());
        float depthRange = maxDepth - minDepth;
        if (depthRange < 1.0f) {
            std::cerr << "Warning: Depth range too small" << std::endl;
            return std::vector<float>(rawDepths.size(), 0.0f);
        }
        for (float depth : rawDepths) {
            if (depth <= 0.0f) {
                normalizedDepths.push_back(0.0f);  // ãƒžã‚¹ã‚¯å¤–ã¯ãƒ•ãƒ©ãƒƒãƒˆ
            } else {
                float normalized = (depth - minDepth) / depthRange;
                normalized = std::max(0.0f, std::min(1.0f, normalized));
                normalizedDepths.push_back(normalized);
            }
        }
        std::cout << "Normalized " << normalizedDepths.size() << " depth values" << std::endl;
        if (maskRatio < 1.0f) {
            normalizedDepths = applyEdgeMask(normalizedDepths, gridWidth, gridHeight, maskRatio);
        }
        std::cout << "Applying Gaussian smoothing (5x5)..." << std::endl;
        normalizedDepths = applyGaussianSmoothing(normalizedDepths, gridWidth, gridHeight, 2);
        std::cout << "Smoothing complete" << std::endl;
        return normalizedDepths;
    }
    std::vector<float> calculateNormalizedDepth(int gridWidth, int gridHeight, float percentile = 0.95f) {
        if (depthImageData.empty()) {
            std::cerr << "Error: No depth image loaded" << std::endl;
            return std::vector<float>();
        }
        int imgWidth = depthWidth;
        int imgHeight = depthHeight;
        float blockWidthF  = (float)imgWidth  / gridWidth;
        float blockHeightF = (float)imgHeight / gridHeight;
        int halfBlockW = std::max(1, (int)(blockWidthF  / 2.0f));
        int halfBlockH = std::max(1, (int)(blockHeightF / 2.0f));
        std::cout << "Block size: " << blockWidthF << "x" << blockHeightF
                  << " pixels (float)" << std::endl;
        std::vector<float> rawDepths;
        rawDepths.reserve((gridWidth + 1) * (gridHeight + 1));
        for (int gy = 0; gy <= gridHeight; gy++) {
            for (int gx = 0; gx <= gridWidth; gx++) {
                int centerX = (int)((float)gx / gridWidth  * (imgWidth  - 1));
                int centerY = (int)((float)gy / gridHeight * (imgHeight - 1));
                int startX = std::max(0, centerX - halfBlockW);
                int endX   = std::min(imgWidth,  centerX + halfBlockW + 1);
                int startY = std::max(0, centerY - halfBlockH);
                int endY   = std::min(imgHeight, centerY + halfBlockH + 1);
                float sum = 0.0f;
                int count = 0;
                for (int y = startY; y < endY; y++) {
                    for (int x = startX; x < endX; x++) {
                        float val = depthImageData[y * depthWidth + x];
                        if (val > 0.0f) {   // ãƒžã‚¹ã‚¯å¤–(æ·±åº¦0)ã‚’é™¤å¤–
                            sum += val;
                            count++;
                        }
                    }
                }
                float avgDepth = (count > 0) ? (sum / count) : 0.0f;
                rawDepths.push_back(avgDepth);
            }
        }
        std::vector<float> sortedDepths = rawDepths;
        std::sort(sortedDepths.begin(), sortedDepths.end());
        int percentileIndex = static_cast<int>(sortedDepths.size() * percentile);
        float maxDepth = sortedDepths[percentileIndex];
        float minDepth = 0.0f;
        for (float d : sortedDepths) {
            if (d > 0.0f) { minDepth = d; break; }
        }
        std::cout << "Raw depth range: " << minDepth << " - " << maxDepth
                  << " (95th percentile, excluding mask)" << std::endl;
        std::vector<float> normalizedDepths;
        normalizedDepths.reserve(rawDepths.size());
        float depthRange = maxDepth - minDepth;
        if (depthRange < 1.0f) {
            std::cerr << "Warning: Depth range too small, using raw values" << std::endl;
            return std::vector<float>(rawDepths.size(), 0.0f);
        }
        for (float depth : rawDepths) {
            if (depth <= 0.0f) {
                normalizedDepths.push_back(0.0f);
            } else {
                float normalized = (depth - minDepth) / depthRange;
                normalized = std::max(0.0f, std::min(1.0f, normalized));
                normalizedDepths.push_back(normalized);
            }
        }
        std::cout << "Normalized " << normalizedDepths.size() << " depth values" << std::endl;
        std::cout << "Applying Gaussian smoothing..." << std::endl;
        normalizedDepths = applyGaussianSmoothing(normalizedDepths, gridWidth, gridHeight, 3);
        std::cout << "Smoothing complete" << std::endl;
        return normalizedDepths;
    }
    void recalculateFrontNormals(int gridWidth, int gridHeight) {
        int frontVertexCount = (gridWidth + 1) * (gridHeight + 1);
        std::cout << "Recalculating normals for " << frontVertexCount << " front vertices..." << std::endl;
        for (int i = 0; i < frontVertexCount * 3; i++) {
            mNormals[i] = 0.0f;
        }
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                GLuint topLeft = y * (gridWidth + 1) + x;
                GLuint topRight = topLeft + 1;
                GLuint bottomLeft = (y + 1) * (gridWidth + 1) + x;
                GLuint bottomRight = bottomLeft + 1;
                glm::vec3 vTL(mVertices[topLeft*3], mVertices[topLeft*3+1], mVertices[topLeft*3+2]);
                glm::vec3 vTR(mVertices[topRight*3], mVertices[topRight*3+1], mVertices[topRight*3+2]);
                glm::vec3 vBL(mVertices[bottomLeft*3], mVertices[bottomLeft*3+1], mVertices[bottomLeft*3+2]);
                glm::vec3 vBR(mVertices[bottomRight*3], mVertices[bottomRight*3+1], mVertices[bottomRight*3+2]);
                glm::vec3 edge1_1 = vBL - vTL;
                glm::vec3 edge2_1 = vTR - vTL;
                glm::vec3 normal1 = glm::normalize(glm::cross(edge1_1, edge2_1));
                glm::vec3 edge1_2 = vBL - vTR;
                glm::vec3 edge2_2 = vBR - vTR;
                glm::vec3 normal2 = glm::normalize(glm::cross(edge1_2, edge2_2));
                mNormals[topLeft*3] += normal1.x;
                mNormals[topLeft*3+1] += normal1.y;
                mNormals[topLeft*3+2] += normal1.z;
                mNormals[bottomLeft*3] += normal1.x;
                mNormals[bottomLeft*3+1] += normal1.y;
                mNormals[bottomLeft*3+2] += normal1.z;
                mNormals[topRight*3] += normal1.x;
                mNormals[topRight*3+1] += normal1.y;
                mNormals[topRight*3+2] += normal1.z;
                mNormals[topRight*3] += normal2.x;
                mNormals[topRight*3+1] += normal2.y;
                mNormals[topRight*3+2] += normal2.z;
                mNormals[bottomLeft*3] += normal2.x;
                mNormals[bottomLeft*3+1] += normal2.y;
                mNormals[bottomLeft*3+2] += normal2.z;
                mNormals[bottomRight*3] += normal2.x;
                mNormals[bottomRight*3+1] += normal2.y;
                mNormals[bottomRight*3+2] += normal2.z;
            }
        }
        for (int i = 0; i < frontVertexCount; i++) {
            glm::vec3 n(mNormals[i*3], mNormals[i*3+1], mNormals[i*3+2]);
            float length = glm::length(n);
            if (length > 0.0001f) {
                n /= length;
            } else {
                n = glm::vec3(0.0f, 0.0f, 1.0f);
            }
            mNormals[i*3] = n.x;
            mNormals[i*3+1] = n.y;
            mNormals[i*3+2] = n.z;
        }
        std::cout << "Normals recalculated." << std::endl;
    }
    void generateGridPlaneWithDepth(int gridWidth, int gridHeight,
                                    const std::vector<float>& normalizedDepths,
                                    float thickness = 0.05f,
                                    float depthScale = 1.0f) {
        mVertices.clear();
        mNormals.clear();
        mTexCoords.clear();
        mIndices.clear();
        float planeWidth = (float)gridWidth / gridHeight;
        float planeHeight = 1.0f;
        float halfThickness = thickness / 2.0f;
        int expectedSize = (gridWidth + 1) * (gridHeight + 1);
        if (normalizedDepths.size() != expectedSize) {
            std::cerr << "Error: Depth data size mismatch. Expected " << expectedSize
                      << ", got " << normalizedDepths.size() << std::endl;
            return;
        }
        std::cout << "Generating depth grid: " << (gridWidth+1) << "x" << (gridHeight+1)
                  << " vertices, depth scale=" << depthScale << std::endl;
        int depthIndex = 0;
        for (int y = 0; y <= gridHeight; y++) {
            for (int x = 0; x <= gridWidth; x++) {
                float u = (float)x / gridWidth;
                float v = (float)y / gridHeight;
                float posX = (u - 0.5f) * planeWidth;
                float posY = (0.5f - v) * planeHeight;
                float depth = normalizedDepths[depthIndex++];
                float posZ = halfThickness + depth * depthScale;
                mVertices.push_back(posX);
                mVertices.push_back(posY);
                mVertices.push_back(posZ);
                mNormals.push_back(0.0f);
                mNormals.push_back(0.0f);
                mNormals.push_back(1.0f);
                mTexCoords.push_back(u);
                mTexCoords.push_back(1.0f - v);
            }
        }
        int frontVertexCount = (gridWidth + 1) * (gridHeight + 1);
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                GLuint topLeft = y * (gridWidth + 1) + x;
                GLuint topRight = topLeft + 1;
                GLuint bottomLeft = (y + 1) * (gridWidth + 1) + x;
                GLuint bottomRight = bottomLeft + 1;
                mIndices.push_back(topLeft);
                mIndices.push_back(bottomLeft);
                mIndices.push_back(topRight);
                mIndices.push_back(topRight);
                mIndices.push_back(bottomLeft);
                mIndices.push_back(bottomRight);
            }
        }
        GLuint backStart = mVertices.size() / 3;
        float backZ = -halfThickness;
        for (int y = 0; y <= gridHeight; y++) {
            for (int x = 0; x <= gridWidth; x++) {
                float u = (float)x / gridWidth;
                float v = (float)y / gridHeight;
                float posX = (u - 0.5f) * planeWidth;
                float posY = (0.5f - v) * planeHeight;
                float posZ = backZ;
                mVertices.push_back(posX);
                mVertices.push_back(posY);
                mVertices.push_back(posZ);
                mNormals.push_back(0.0f);
                mNormals.push_back(0.0f);
                mNormals.push_back(-1.0f);
                mTexCoords.push_back(-1.0f);
                mTexCoords.push_back(-1.0f);
            }
        }
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                GLuint topLeft = backStart + y * (gridWidth + 1) + x;
                GLuint topRight = topLeft + 1;
                GLuint bottomLeft = backStart + (y + 1) * (gridWidth + 1) + x;
                GLuint bottomRight = bottomLeft + 1;
                mIndices.push_back(topLeft);
                mIndices.push_back(topRight);
                mIndices.push_back(bottomLeft);
                mIndices.push_back(topRight);
                mIndices.push_back(bottomRight);
                mIndices.push_back(bottomLeft);
            }
        }
        for (int x = 0; x < gridWidth; x++) {
            GLuint frontBL = x;
            GLuint frontBR = x + 1;
            GLuint backBL = backStart + x;
            GLuint backBR = backStart + x + 1;
            mIndices.push_back(frontBL);
            mIndices.push_back(backBL);
            mIndices.push_back(frontBR);
            mIndices.push_back(frontBR);
            mIndices.push_back(backBL);
            mIndices.push_back(backBR);
        }
        for (int x = 0; x < gridWidth; x++) {
            GLuint frontTL = gridHeight * (gridWidth + 1) + x;
            GLuint frontTR = frontTL + 1;
            GLuint backTL = backStart + gridHeight * (gridWidth + 1) + x;
            GLuint backTR = backTL + 1;
            mIndices.push_back(frontTL);
            mIndices.push_back(frontTR);
            mIndices.push_back(backTL);
            mIndices.push_back(frontTR);
            mIndices.push_back(backTR);
            mIndices.push_back(backTL);
        }
        for (int y = 0; y < gridHeight; y++) {
            GLuint frontTL = y * (gridWidth + 1);
            GLuint frontBL = (y + 1) * (gridWidth + 1);
            GLuint backTL = backStart + y * (gridWidth + 1);
            GLuint backBL = backStart + (y + 1) * (gridWidth + 1);
            mIndices.push_back(frontTL);
            mIndices.push_back(frontBL);
            mIndices.push_back(backTL);
            mIndices.push_back(frontBL);
            mIndices.push_back(backBL);
            mIndices.push_back(backTL);
        }
        for (int y = 0; y < gridHeight; y++) {
            GLuint frontTR = y * (gridWidth + 1) + gridWidth;
            GLuint frontBR = (y + 1) * (gridWidth + 1) + gridWidth;
            GLuint backTR = backStart + y * (gridWidth + 1) + gridWidth;
            GLuint backBR = backStart + (y + 1) * (gridWidth + 1) + gridWidth;
            mIndices.push_back(frontTR);
            mIndices.push_back(backTR);
            mIndices.push_back(frontBR);
            mIndices.push_back(frontBR);
            mIndices.push_back(backTR);
            mIndices.push_back(backBR);
        }
        numFaces = mIndices.size() / 3;
        recalculateFrontNormals(gridWidth, gridHeight);
        std::cout << "Generated: " << (mVertices.size()/3) << " vertices, "
                  << numFaces << " triangles" << std::endl;
    }
    /**
         */
    void applyTransformation(const glm::mat4& T) {
        int vertexCount = mVertices.size() / 3;
        if (vertexCount == 0) {
            std::cerr << "Warning: No vertices to transform" << std::endl;
            return;
        }
        std::cout << "Applying transformation to " << vertexCount << " vertices..." << std::endl;
        for (int i = 0; i < vertexCount; i++) {
            glm::vec4 vertex(
                mVertices[i * 3 + 0],
                mVertices[i * 3 + 1],
                mVertices[i * 3 + 2],
                1.0f
                );
            glm::vec4 transformedVertex = T * vertex;
            mVertices[i * 3 + 0] = transformedVertex.x;
            mVertices[i * 3 + 1] = transformedVertex.y;
            mVertices[i * 3 + 2] = transformedVertex.z;
        }
        if (!mNormals.empty() && mNormals.size() == mVertices.size()) {
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(T)));
            for (int i = 0; i < vertexCount; i++) {
                glm::vec3 normal(
                    mNormals[i * 3 + 0],
                    mNormals[i * 3 + 1],
                    mNormals[i * 3 + 2]
                    );
                glm::vec3 transformedNormal = normalMatrix * normal;
                transformedNormal = glm::normalize(transformedNormal);
                mNormals[i * 3 + 0] = transformedNormal.x;
                mNormals[i * 3 + 1] = transformedNormal.y;
                mNormals[i * 3 + 2] = transformedNormal.z;
            }
        }
        updateVBO();
        std::cout << "Transformation applied successfully." << std::endl;
    }
    glm::vec3 calcCenter() const {
        glm::vec3 center(0.0f);
        size_t vertexCount = mVertices.size() / 3;
        if (vertexCount == 0) return center;
        for (size_t i = 0; i < mVertices.size(); i += 3) {
            center.x += mVertices[i];
            center.y += mVertices[i + 1];
            center.z += mVertices[i + 2];
        }
        center /= static_cast<float>(vertexCount);
        return center;
    }

    void translate(const glm::vec3& direction) {
        for (size_t i = 0; i < mVertices.size(); i += 3) {
            mVertices[i]     += direction.x;
            mVertices[i + 1] += direction.y;
            mVertices[i + 2] += direction.z;
        }
    }

    void scaleAround(const glm::vec3& center, float factor) {
        for (size_t i = 0; i < mVertices.size(); i += 3) {
            mVertices[i]     = (mVertices[i]     - center.x) * factor + center.x;
            mVertices[i + 1] = (mVertices[i + 1] - center.y) * factor + center.y;
            mVertices[i + 2] = (mVertices[i + 2] - center.z) * factor + center.z;
        }
    }

    void rotateAround(const glm::vec3& center,
                      const glm::vec3& axis1, float angle1,
                      const glm::vec3& axis2, float angle2) {
        glm::mat4 T = glm::mat4(1.0f);
        T = glm::translate(T, center);
        T = glm::rotate(T, angle1, axis1);
        T = glm::rotate(T, angle2, axis2);
        T = glm::translate(T, -center);
        for (size_t i = 0; i < mVertices.size(); i += 3) {
            glm::vec4 v(mVertices[i], mVertices[i + 1], mVertices[i + 2], 1.0f);
            v = T * v;
            mVertices[i]     = v.x;
            mVertices[i + 1] = v.y;
            mVertices[i + 2] = v.z;
        }
    }

    void updateVBO() {
        if (VAO == 0) {
            std::cerr << "Warning: VAO not initialized, skipping VBO update" << std::endl;
            return;
        }
        glBindVertexArray(VAO);
        if (VBO != 0 && !mVertices.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(
                GL_ARRAY_BUFFER,
                0,
                mVertices.size() * sizeof(GLfloat),
                mVertices.data()
                );
        }
        if (NBO != 0 && !mNormals.empty()) {
            glBindBuffer(GL_ARRAY_BUFFER, NBO);
            glBufferSubData(
                GL_ARRAY_BUFFER,
                0,
                mNormals.size() * sizeof(GLfloat),
                mNormals.data()
                );
        }
        glBindVertexArray(0);
    }
};
void setUp(mCutMesh& srcMesh) {
    while (glGetError() != GL_NO_ERROR) {}
    if (srcMesh.mVertices.empty() || srcMesh.mIndices.empty()) {
        std::cerr << "Error: Empty mesh data" << std::endl;
        return;
    }

    // ★ 既存のGLリソースを解放してからリーク防止
    if (srcMesh.VAO != 0) {
        glDeleteVertexArrays(1, &srcMesh.VAO);
        srcMesh.VAO = 0;
    }
    if (srcMesh.VBO != 0) {
        glDeleteBuffers(1, &srcMesh.VBO);
        srcMesh.VBO = 0;
    }
    if (srcMesh.EBO != 0) {
        glDeleteBuffers(1, &srcMesh.EBO);
        srcMesh.EBO = 0;
    }
    if (srcMesh.NBO != 0) {
        glDeleteBuffers(1, &srcMesh.NBO);
        srcMesh.NBO = 0;
    }
    if (srcMesh.TBO != 0) {
        glDeleteBuffers(1, &srcMesh.TBO);
        srcMesh.TBO = 0;
    }

    std::vector<float> vertices(srcMesh.mVertices.size());
    std::vector<GLuint> indices(srcMesh.mIndices.size());
    for(size_t i = 0; i < srcMesh.mVertices.size(); i++) {
        vertices[i] = srcMesh.mVertices[i];
    }
    for(size_t i = 0; i < srcMesh.mIndices.size(); i++) {
        indices[i] = static_cast<GLuint>(srcMesh.mIndices[i]);
    }
    size_t vertexCount = vertices.size() / 3;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= vertexCount) {
            std::cerr << "Error: Index out of range at " << i << ": " << indices[i]
                      << " (vertex count: " << vertexCount << ")" << std::endl;
            return;
        }
    }
    std::vector<float> normals(vertices.size(), 0.0f);
    if (!srcMesh.mNormals.empty() && srcMesh.mNormals.size() == vertices.size()) {
        normals = srcMesh.mNormals;
    } else {
        for (size_t i = 0; i < indices.size(); i += 3) {
            GLuint i0 = indices[i];
            GLuint i1 = indices[i + 1];
            GLuint i2 = indices[i + 2];
            glm::vec3 v0(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
            glm::vec3 v1(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
            glm::vec3 v2(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
            for (GLuint idx : {i0, i1, i2}) {
                normals[idx * 3]     += normal.x;
                normals[idx * 3 + 1] += normal.y;
                normals[idx * 3 + 2] += normal.z;
            }
        }
        for (size_t i = 0; i < normals.size(); i += 3) {
            glm::vec3 n(normals[i], normals[i + 1], normals[i + 2]);
            float length = glm::length(n);
            if (length > 0.0001f) {
                n /= length;
            } else {
                n = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            normals[i]     = n.x;
            normals[i + 1] = n.y;
            normals[i + 2] = n.z;
        }
    }
    glGenVertexArrays(1, &srcMesh.VAO);
    glGenBuffers(1, &srcMesh.VBO);
    glGenBuffers(1, &srcMesh.EBO);
    glGenBuffers(1, &srcMesh.NBO);
    glGenBuffers(1, &srcMesh.TBO);
    glBindVertexArray(srcMesh.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, srcMesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, srcMesh.NBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(GLfloat), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);
    if (!srcMesh.mTexCoords.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, srcMesh.TBO);
        glBufferData(GL_ARRAY_BUFFER, srcMesh.mTexCoords.size() * sizeof(GLfloat),
                     srcMesh.mTexCoords.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
        glEnableVertexAttribArray(2);
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, srcMesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}
#endif
