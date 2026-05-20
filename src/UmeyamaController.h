#pragma once
// UmeyamaController.h
// 2画面分割 Umeyama Manual Registration の全ロジック。
// main.cpp はインスタンス1個＋フック数行のみ。

#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>

#include "mCutMesh.h"
#include "MeshDrawing.h"
#include "RegistrationCore.h"
#include "RegistrationUI.h"
#include "RayCast.h"
#include "Sphere.h"
#include "FullSphereCameraWithTarget.h"

// main.cpp グローバル（最小限の extern）
extern int gWindowWidth, gWindowHeight;
extern glm::mat4 view, projection;
extern glm::vec3 objPos;
extern float g_sceneDiag;   // for scene-scale-relative marker radius (sync with RegRatios)

struct UmeyamaController {

    // ===================== 内部状態 =====================
    FullSphereCamera camLeft;       // 左画面: liver (ソース)
    FullSphereCamera camRight;      // 右画面: board (ターゲットOBJ)
    bool active = false;            // 2画面モード ON/OFF
    SphereMesh marker;              // 対応点の球マーカー
    bool markerReady = false;

    // start()前のメインカメラintrinsicsを保存（execute/cancelで復元用）
    float savedFx = 0, savedFy = 0, savedCx = 0, savedCy = 0;
    int   savedCalibW = 0, savedCalibH = 0;
    bool  savedUseIntrinsics = false;

    // ===================== 公開メソッド =====================

    // GL初期化後に1回呼ぶ
    void init() {
        marker.generate(1.0f, 12, 12);
        marker.setup();
        markerReady = true;
    }

    // Umeyamaボタン押下時 — 2画面モード開始
    void start(RegistrationData& reg, const FullSphereCamera& mainCam,
               int winW, int winH) {
        reg.reset();
        reg.targetPointCount = 5;
        reg.state = RegistrationData::SELECTING_BOARD_POINTS;
        reg.useRegistration = false;

        // メインカメラのintrinsicsを保存（execute/cancelで復元）
        savedFx = mainCam.fx;  savedFy = mainCam.fy;
        savedCx = mainCam.cx;  savedCy = mainCam.cy;
        savedCalibW = mainCam.calibWidth;
        savedCalibH = mainCam.calibHeight;
        savedUseIntrinsics = mainCam.useIntrinsics;

        // 左右カメラをメインカメラからコピー
        camLeft  = mainCam;
        camRight = mainCam;

        // intrinsics保護を無効化（分割ビューポートではFOVベースにする）
        camLeft.calibWidth = 0;  camLeft.calibHeight = 0;
        camLeft.useIntrinsics = false;
        camRight.calibWidth = 0; camRight.calibHeight = 0;
        camRight.useIntrinsics = false;

        int halfW = winW / 2;
        camLeft.currentTarget  = TARGET_LIVER;
        camLeft.gRadius        = mainCam.InitialRadius;
        camLeft.cx             = halfW / 2.0f;
        camLeft.cy             = winH / 2.0f;

        camRight.currentTarget = TARGET_TEXTURE;
        camRight.gRadius       = mainCam.InitialRadius * 2.0f;
        camRight.cx            = halfW / 2.0f;
        camRight.cy            = winH / 2.0f;

        active = true;
        std::cout << "[Umeyama] Split screen started. "
                  << "RIGHT: select " << reg.targetPointCount
                  << " board points" << std::endl;
    }

    // intrinsicsをメインカメラに復元
    void restoreIntrinsics(FullSphereCamera& mainCam) {
        mainCam.fx = savedFx;  mainCam.fy = savedFy;
        mainCam.cx = savedCx;  mainCam.cy = savedCy;
        mainCam.calibWidth  = savedCalibW;
        mainCam.calibHeight = savedCalibH;
        mainCam.useIntrinsics = savedUseIntrinsics;
    }

    // Executeボタン — 変換適用＋1画面復帰
    void execute(RegistrationData& reg, std::vector<mCutMesh*>& organs,
                 FullSphereCamera& mainCam, int winW, int winH) {
        if (!reg.canRegister()) return;

        performRegistrationUmeyama(reg, organs);

        // 2画面を閉じてメインカメラに復帰
        active = false;

        // camRightの回転・ターゲット位置だけ引き継ぎ、intrinsicsは復元
        mainCam.rotation      = camRight.rotation;
        mainCam.currentTarget = TARGET_TEXTURE;
        mainCam.gRadius       = mainCam.InitialRadius;
        restoreIntrinsics(mainCam);

        std::cout << "[Umeyama] Registration executed. Returning to single view."
                  << std::endl;
    }

    // Undoボタン
    void undoPoint(RegistrationData& reg) {
        if (reg.state == RegistrationData::READY_TO_REGISTER ||
            reg.state == RegistrationData::SELECTING_OBJECT_POINTS) {
            if (!reg.objectPoints.empty()) {
                reg.objectPoints.pop_back();
                std::cout << "[Umeyama] Undo object point. Remaining: "
                          << reg.objectPoints.size() << std::endl;
                if (reg.state == RegistrationData::READY_TO_REGISTER)
                    reg.state = RegistrationData::SELECTING_OBJECT_POINTS;
                return;
            }
        }
        if (reg.state == RegistrationData::SELECTING_OBJECT_POINTS &&
            reg.objectPoints.empty()) {
            if (!reg.boardPoints.empty()) {
                reg.boardPoints.pop_back();
                reg.state = RegistrationData::SELECTING_BOARD_POINTS;
                std::cout << "[Umeyama] Undo board point (back to board). Remaining: "
                          << reg.boardPoints.size() << std::endl;
            }
            return;
        }
        if (reg.state == RegistrationData::SELECTING_BOARD_POINTS) {
            if (!reg.boardPoints.empty()) {
                reg.boardPoints.pop_back();
                std::cout << "[Umeyama] Undo board point. Remaining: "
                          << reg.boardPoints.size() << std::endl;
            }
        }
    }

    // Cancelボタン
    void cancel(RegistrationData& reg, FullSphereCamera& mainCam) {
        active = false;
        reg.reset();
        restoreIntrinsics(mainCam);
        std::cout << "[Umeyama] Cancelled. Returned to single view." << std::endl;
    }

    // マウスクリック — ポイントピッキング
    // 戻り値: true = イベント消費（通常処理をスキップ）
    bool handleMouse(float x, float y, int button, int winW, int winH,
                     RegistrationData& reg, mCutMesh* liver, mCutMesh* screen) {
        if (!active) return false;
        if (button != 0) return true;  // 左クリックのみ、他はブロック

        int halfW = winW / 2;
        bool isRight = (x >= halfW);

        // Phase 1: board point 選択（右画面のみ受付）
        if (reg.state == RegistrationData::SELECTING_BOARD_POINTS) {
            if (!isRight) {
                std::cout << "[Umeyama] Board point: click on RIGHT screen" << std::endl;
                return true;
            }
            float localX = x - halfW;
            glm::vec3 pt = pickOnMesh(localX, y, screen, &camRight, halfW, winH);
            if (pt.x > -900.0f) {
                reg.boardPoints.push_back(pt);
                int idx = (int)reg.boardPoints.size() - 1;
                std::cout << "[Umeyama] Board #" << (idx+1)
                          << "/" << reg.targetPointCount
                          << " at (" << pt.x << "," << pt.y << "," << pt.z << ")"
                          << std::endl;
                if ((int)reg.boardPoints.size() >= reg.targetPointCount) {
                    reg.state = RegistrationData::SELECTING_OBJECT_POINTS;
                    std::cout << "[Umeyama] >> Now select liver points on LEFT screen"
                              << std::endl;
                }
            }
            return true;
        }

        // Phase 2: object point 選択（左画面のみ受付）
        if (reg.state == RegistrationData::SELECTING_OBJECT_POINTS) {
            if (isRight) {
                std::cout << "[Umeyama] Object point: click on LEFT screen" << std::endl;
                return true;
            }
            glm::vec3 pt = pickOnMesh(x, y, liver, &camLeft, halfW, winH);
            if (pt.x > -900.0f) {
                reg.objectPoints.push_back(pt);
                int idx = (int)reg.objectPoints.size() - 1;
                std::cout << "[Umeyama] Object #" << (idx+1)
                          << " at (" << pt.x << "," << pt.y << "," << pt.z << ")"
                          << std::endl;
                if (reg.objectPoints.size() >= reg.boardPoints.size()) {
                    reg.state = RegistrationData::READY_TO_REGISTER;
                    std::cout << "[Umeyama] >> All points selected. Press Execute."
                              << std::endl;
                }
            }
            return true;
        }

        return true;  // active中は他のマウス操作をブロック
    }

    // マウスドラッグ — 左右独立カメラ操作
    void handleMouseMove(float x, float dx, float dy,
                         bool leftBtn, bool rightBtn, int winW) {
        if (!active) return;
        FullSphereCamera& cam = (x < winW / 2.0f) ? camLeft : camRight;
        if (leftBtn && !rightBtn) cam.Rotate(dx, dy);
        if (rightBtn && !leftBtn) cam.Pan(dx, -dy);
    }

    // マウスホイール — ズーム
    void handleScroll(float x, float deltaY, int winW) {
        if (!active) return;
        FullSphereCamera& cam = (x < winW / 2.0f) ? camLeft : camRight;
        cam.gRadius += (float)deltaY * cam.ZOOM_SENSITIVITY;
        cam.gRadius  = glm::clamp(cam.gRadius, 2.0f, 80.0f);
    }

    // 2画面描画
    void render(ShaderProgram& shader, ShaderProgram& texShader,
                const RegistrationData& reg,
                const std::vector<mCutMesh*>& organs,
                mCutMesh* screen, int winW, int winH) {
        if (!active) return;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        int halfW = winW / 2;
        glm::mat4 mdl = glm::translate(glm::mat4(1.0f), objPos);

        // ======== 左画面: liver（ソース）========
        {
            glViewport(0, 0, halfW, winH);

            // liver中心にカメラを向ける
            glm::vec3 liverC(0.0f);
            if (!organs.empty() && organs[0]) {
                liverC = FullSphereCamera::calculateMeshCenter(organs[0]->mVertices);
            }
            camLeft.updateTargetPositions(liverC, glm::vec3(0));
            camLeft.UpdateCamera();

            glm::mat4 leftView = camLeft.view;
            glm::mat4 leftProj = createProjectionForViewport(halfW, winH, camLeft);

            // 臓器メッシュ描画
            std::vector<mCutMesh*> visMeshes;
            std::vector<glm::vec4> visColors;
            const glm::vec4 defCol[] = {
                {0.8f,0.2f,0.2f,0.8f}, {0.2f,0.2f,0.8f,0.8f},
                {0.2f,0.5f,0.5f,0.8f}, {0.8f,0.5f,0.5f,0.8f},
                {0.2f,0.8f,0.5f,0.6f}, {0.2f,0.8f,0.2f,0.8f},
            };
            for (size_t i = 0; i < organs.size() && i < 6; i++) {
                if (organs[i] && !organs[i]->mVertices.empty()) {
                    visMeshes.push_back(organs[i]);
                    visColors.push_back(defCol[i]);
                }
            }
            draw_AllmCutMeshes(visMeshes, shader, texShader,
                               camLeft.cameraPos, visColors,
                               mdl, leftView, leftProj, -1);

            // object point マーカー
            drawMarkers(shader, reg, leftView, leftProj, camLeft.cameraPos, false);
        }

        // ======== 右画面: board（ターゲットOBJ）========
        {
            glViewport(halfW, 0, halfW, winH);

            // OBJ中心にカメラを向ける
            glm::vec3 screenC(0.0f);
            if (screen)
                screenC = FullSphereCamera::calculateMeshCenter(screen->mVertices);
            camRight.updateTargetPositions(glm::vec3(0), screenC);
            camRight.UpdateCamera();

            glm::mat4 rightView = camRight.view;
            glm::mat4 rightProj = createProjectionForViewport(halfW, winH, camRight);

            // screenMeshを通常メッシュとして描画（テクスチャ非依存）
            std::vector<mCutMesh*> rightMeshes = { screen };
            std::vector<glm::vec4> rightColors = { {0.3f, 0.6f, 0.9f, 0.5f} };
            draw_AllmCutMeshes(rightMeshes, shader, texShader,
                               camRight.cameraPos, rightColors,
                               mdl, rightView, rightProj, -1);

            // board point マーカー
            drawMarkers(shader, reg, rightView, rightProj, camRight.cameraPos, true);
        }

        // ビューポート復元
        glViewport(0, 0, winW, winH);
    }

private:
    // レイキャストでメッシュ上のクリック位置を取得
    glm::vec3 pickOnMesh(float sx, float sy, mCutMesh* mesh,
                         FullSphereCamera* cam, int vpW, int vpH) {
        const glm::vec3 MISS(-999.0f);
        if (!mesh || mesh->mVertices.empty()) return MISS;

        cam->UpdateCamera();
        glm::mat4 v = cam->view;
        glm::mat4 p = createProjectionForViewport(vpW, vpH, *cam);

        RayCast::Ray ray = RayCast::screenToRay(
            sx, sy, v, p, glm::vec4(0, 0, vpW, vpH));
        RayCast::RayHitTri hit = RayCast::intersectMesh(
            ray, mesh->mVertices, mesh->mIndices);

        return hit.hit ? hit.position : MISS;
    }

    // 対応点の球マーカー描画
    void drawMarkers(ShaderProgram& shader, const RegistrationData& reg,
                     const glm::mat4& v, const glm::mat4& p,
                     const glm::vec3& camPos, bool boardSide) {
        if (!markerReady) return;
        // Marker radius = kRefMarkerCorrespondence (0.30) * scaleRatio.
        // Kept in sync with RegRatios::markerCorrespondence() in RegistrationActions.h.
        // (Hardcoded here to avoid pulling RegistrationActions.h into this header.)
        constexpr float kRefMarker      = 0.30f;
        constexpr float kRefSceneDiag   = 7.36f;
        const float radius = kRefMarker * (g_sceneDiag / kRefSceneDiag);
        const auto& pts = boardSide ? reg.boardPoints : reg.objectPoints;
        for (size_t i = 0; i < pts.size(); i++) {
            glm::vec3 color = getPointColor((int)i, boardSide);
            marker.draw(shader, pts[i], color, radius, v, p, camPos);
        }
    }
};
