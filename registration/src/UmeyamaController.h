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
#include "SoftPartition.h"   // Phase U-1

// main.cpp グローバル（最小限の extern）
extern int gWindowWidth, gWindowHeight;
extern glm::mat4 view, projection;
extern glm::vec3 objPos;
extern float g_sceneDiag;   // for scene-scale-relative marker radius (sync with RegRatios)

// Phase U-1: main.cpp で定義、execute() が書き込む
extern SoftPartition::AnchorSet g_softAnchors;
extern float                    g_softUmeyamaScale;
extern bool                     g_softPartReady;

// boardMesh3D: 右画面でテクスチャ付き板メッシュを screenMesh (target cloud)
// と一緒に表示するために参照する。
//   - マーキング時に board の絵柄と target の対応関係が一目で分かるので
//     クリック対応点を取りやすくなる、というユーザー要望に対応。
extern mCutMesh* boardMesh3D;
extern float     g_meshAlpha[8];  // [6]=boardMesh, [7]=screenMesh の alpha

// screenMesh 点群描画フラグと描画関数 (main.cpp で定義)。
//   Umeyama 2画面モード中、右ビューポートで targetCloud (= screenMesh) を
//   ユーザーの "Draw as points" / Density 設定に従って軽量描画するために
//   ここから呼ぶ。これをやらずに draw_AllmCutMeshes で full-triangle 描画
//   すると数十万頂点メッシュでフレームレートが大幅低下する。
extern bool  g_screenMeshAsPoints;
extern float g_screenMeshPointSize;
extern float g_screenMeshDensity;
void drawScreenMeshAsPoints(
    mCutMesh* mesh,
    ShaderProgram& shader,
    const glm::mat4& model,
    const glm::mat4& view,
    const glm::mat4& projection,
    const glm::vec3& camPos,
    const glm::vec4& color,
    float pointSize);

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

    // ----- mesh center cache (chat: Umeyama frame-rate fix) ---------------
    // Umeyama 2画面モードはユーザーが対応点をクリックするだけで、organs と
    // screen mesh は active=true の間まったく動かない (registration apply は
    // execute() で active=false になった後にしか実行されない)。
    // それなのに従来は毎フレーム O(N) の calculateMeshCenter を 2 回走らせて
    // いた (liver 9,992 verts + screen 数十万 verts)。これがフレームレート
    // ドロップの主犯候補なので start() で一度だけ計算してキャッシュする。
    // ---------------------------------------------------------------------
    glm::vec3 cachedLiverCenter  = glm::vec3(0.0f);
    glm::vec3 cachedScreenCenter = glm::vec3(0.0f);
    bool centersCached           = false;

    // ===================== 公開メソッド =====================

    // GL初期化後に1回呼ぶ
    void init() {
        marker.generate(1.0f, 12, 12);
        marker.setup();
        markerReady = true;
    }

    // Umeyamaボタン押下時 — 2画面モード開始
    void start(RegistrationData& reg, const FullSphereCamera& mainCam,
               int winW, int winH, int pointCount = 3) {
        reg.reset();
        reg.targetPointCount = (pointCount < 3) ? 3 : (pointCount > 5 ? 5 : pointCount);
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
        // 次の render() 呼び出し時に 1 回だけ center を計算するためフラグを下ろす
        centersCached = false;
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

        // ----- Phase U-1: Umeyama 実行前に anchor を捕捉 -----
        // applyRegistrationToMesh が直後に objectPoints を world 座標へ
        // in-place 書き換えするため、local 座標はここで保存する。
        {
            const int N = (int)reg.objectPoints.size();
            if (N >= SoftPartition::MIN_GROUPS && N <= SoftPartition::MAX_GROUPS &&
                (int)reg.objectVertIdx.size() == N &&
                (int)reg.boardPoints.size()   == N) {
                g_softAnchors.clear();
                g_softAnchors.srcVertIdx = reg.objectVertIdx;
                g_softAnchors.srcLocal   = reg.objectPoints;   // LOCAL (まだ未変換)
                g_softAnchors.tgtWorld   = reg.boardPoints;     // world (固定)
                glm::mat4 T_ume = Reg3D::UmeyamaRegistration(
                    reg.objectPoints, reg.boardPoints);
                g_softUmeyamaScale = SoftPartition::extractScaleFromUmeyama(T_ume);
                g_softPartReady = false;   // Shift+U で遅延計算
                std::cout << "[Umeyama] Soft anchor captured: N=" << N
                          << "  s_ume=" << g_softUmeyamaScale << std::endl;
            } else {
                g_softAnchors.clear();
                g_softPartReady = false;
                std::cout << "[Umeyama] Soft anchor NOT captured "
                          << "(N=" << N << ", vertIdx="
                          << reg.objectVertIdx.size() << ")" << std::endl;
            }
        }

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
                if (!reg.objectVertIdx.empty())          // Phase U-1
                    reg.objectVertIdx.pop_back();
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
            int hitVert = -1;
            glm::vec3 pt = pickOnMesh(x, y, liver, &camLeft, halfW, winH, &hitVert);
            if (pt.x > -900.0f) {
                reg.objectPoints.push_back(pt);
                reg.objectVertIdx.push_back(hitVert);     // Phase U-1: 並行保存
                int idx = (int)reg.objectPoints.size() - 1;
                std::cout << "[Umeyama] Object #" << (idx+1)
                          << " at (" << pt.x << "," << pt.y << "," << pt.z << ")"
                          << "  vertIdx=" << hitVert << std::endl;
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

        // ----- mesh centers: compute once per session, then reuse -------
        // organs[0] (liver) と screen はこの session の間動かないので毎フレーム
        // O(N) を走らせる必要はない。start() で centersCached=false を立てたあと
        // 最初の render で 1 回だけ計算してキャッシュする。
        // (ガード: organs/screen が空のときはゼロベクトルのまま、次フレームで再試行)
        if (!centersCached) {
            bool gotLiver = false, gotScreen = false;
            if (!organs.empty() && organs[0] && !organs[0]->mVertices.empty()) {
                cachedLiverCenter =
                    FullSphereCamera::calculateMeshCenter(organs[0]->mVertices);
                gotLiver = true;
            }
            if (screen && !screen->mVertices.empty()) {
                cachedScreenCenter =
                    FullSphereCamera::calculateMeshCenter(screen->mVertices);
                gotScreen = true;
            }
            if (gotLiver && gotScreen) centersCached = true;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        int halfW = winW / 2;
        glm::mat4 mdl = glm::translate(glm::mat4(1.0f), objPos);

        // ======== 左画面: liver（ソース）========
        {
            glViewport(0, 0, halfW, winH);

            // liver中心にカメラを向ける (キャッシュ済み)
            camLeft.updateTargetPositions(cachedLiverCenter, glm::vec3(0));
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

        // ======== 右画面: board (テクスチャ付き板) + target (screenMesh) ========
        //
        // マーキングしやすさのため、通常モードと同じレイアウト:
        //   board (テクスチャ付き、白) を下地に描画
        //   target (screenMesh) を点群 or 半透明三角形で上塗り
        //   board の表示判定は g_meshAlpha[6]、target は g_meshAlpha[7]
        //
        // Umeyama 入口時点で board の alpha がゼロでも右画面では強制的に
        // 表示する (絵柄が見えないとマーキングできないため)。target も同様に
        // 最低限の不透明度を保証する。
        {
            glViewport(halfW, 0, halfW, winH);

            // OBJ中心にカメラを向ける (キャッシュ済み)
            camRight.updateTargetPositions(glm::vec3(0), cachedScreenCenter);
            camRight.UpdateCamera();

            glm::mat4 rightView = camRight.view;
            glm::mat4 rightProj = createProjectionForViewport(halfW, winH, camRight);

            // (a) board (テクスチャ付き) を draw_AllmCutMeshes で描画。
            //     通常モードと同じく textureMeshIdx 指定でテクスチャシェーダに
            //     切り替えてもらう。alpha は Umeyama 中だけ 1.0 に強制 (元値に
            //     関わらずマーキングしやすいように)。
            std::vector<mCutMesh*> rightMeshes;
            std::vector<glm::vec4> rightColors;
            int textureMeshIdx = -1;

            // target を triangle 描画にする場合のみ draw_AllmCutMeshes 側に積む。
            // 点群モードのときは後段で別途 drawScreenMeshAsPoints する。
            if (screen && !screen->mVertices.empty() && !g_screenMeshAsPoints) {
                rightMeshes.push_back(screen);
                rightColors.push_back(glm::vec4(0.3f, 0.6f, 0.9f, 0.5f));
            }

            if (boardMesh3D && !boardMesh3D->mVertices.empty()) {
                rightMeshes.push_back(boardMesh3D);
                rightColors.push_back(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
                textureMeshIdx = (int)rightMeshes.size() - 1;
            }

            if (!rightMeshes.empty()) {
                draw_AllmCutMeshes(rightMeshes, shader, texShader,
                                   camRight.cameraPos, rightColors,
                                   mdl, rightView, rightProj, textureMeshIdx);
            }

            // (b) target を点群で描画 (デフォルト)。board の上に乗るので
            //     density スライダの値次第ではテクスチャがほぼ見えるようになる。
            if (g_screenMeshAsPoints && screen && !screen->mVertices.empty()) {
                drawScreenMeshAsPoints(
                    screen, shader,
                    mdl, rightView, rightProj, camRight.cameraPos,
                    glm::vec4(0.3f, 0.6f, 0.9f, 0.7f),
                    g_screenMeshPointSize);
            }

            // board point マーカー
            drawMarkers(shader, reg, rightView, rightProj, camRight.cameraPos, true);
        }

        // ビューポート復元
        glViewport(0, 0, winW, winH);
    }

private:
    // レイキャストでメッシュ上のクリック位置を取得
    // vertIdxOut (optional): ヒット頂点 index を返す。RayCast はグローバル
    // hit_index にヒット三角形 (i/3) を書き込むので、その 3 頂点から hit.position
    // に最も近い 1 頂点を選ぶ。三角形が取れなければ global nearest で fallback。
    glm::vec3 pickOnMesh(float sx, float sy, mCutMesh* mesh,
                         FullSphereCamera* cam, int vpW, int vpH,
                         int* vertIdxOut = nullptr) {
        const glm::vec3 MISS(-999.0f);
        if (vertIdxOut) *vertIdxOut = -1;
        if (!mesh || mesh->mVertices.empty()) return MISS;

        cam->UpdateCamera();
        glm::mat4 v = cam->view;
        glm::mat4 p = createProjectionForViewport(vpW, vpH, *cam);

        RayCast::Ray ray = RayCast::screenToRay(
            sx, sy, v, p, glm::vec4(0, 0, vpW, vpH));
        RayCast::RayHit hit = RayCast::intersectMesh(
            ray, mesh->mVertices, mesh->mIndices);
        if (!hit.hit) return MISS;

        if (vertIdxOut) {
            int tri  = hit_index;                  // global, set by intersectMesh
            int best = -1;
            if (tri >= 0 && (size_t)(tri * 3 + 2) < mesh->mIndices.size()) {
                float bd = 1e30f;
                for (int k = 0; k < 3; ++k) {
                    int vi = (int)mesh->mIndices[tri * 3 + k];
                    glm::vec3 pv = SoftPartition::vertexAt(*mesh, vi);
                    float d = glm::dot(pv - hit.position, pv - hit.position);
                    if (d < bd) { bd = d; best = vi; }
                }
            }
            if (best < 0)
                best = SoftPartition::nearestVertexIdx(*mesh, hit.position);
            *vertIdxOut = best;
        }
        return hit.position;
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
