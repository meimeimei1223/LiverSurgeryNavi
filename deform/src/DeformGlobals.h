// DeformGlobals.h
// 変形パイプライン用のグローバル変数を集約するヘッダ。
// C++17 の inline 変数で .cpp なしで定義する。
#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "SoftBody.h"
#include "Sphere.h"      // SphereMesh
#include "mCutMesh.h"    // Target / Board 用
#include "Grabber.h"     // bunnyPos の宣言は Grabber.h 経由(extern)→こちらで定義
#include "FullSphereCameraWithTarget.h"   // FullSphereCamera (gOrbitCamPtr 用)

// AutoDeform パイプライン
#include "AutoDeform.h"
#include "AutoHandleController.h"

// ============================================================================
// メインモード(将来 REGISTRATION_MODE と切り替えるため列挙だけ用意)
// ============================================================================
enum MainMode {
    REGISTRATION_MODE,
    DEFORM_MODE
};

// ============================================================================
// 変形サブステート
// ============================================================================
struct DeformHandlPlaceData {
    enum State {
        RIGID_MODE,
        HANDLE_PLACE_MODE,
        DEFORM_MODE,
        PLANECUT_MODE
    };

    State state = RIGID_MODE;
    std::vector<glm::vec3> softbodyPoints;

    void reset();   // multiBody->fullReset() を呼ぶので DeformPipeline.h で定義
};

// ============================================================================
// グローバル(inline 定義 = .cpp 不要)
// ============================================================================
inline SoftBody*           multiBody         = nullptr;
inline Grabber*            gGrabber          = nullptr;
inline FullSphereCamera*   gOrbitCamPtr      = nullptr;   // main.cpp の OrbitCam を指す
inline bool                deformInit        = false;

inline glm::vec3           bunnyPos          = glm::vec3(0.0f);   // Grabber.h で extern 宣言
inline MainMode            currentMainMode   = DEFORM_MODE;

inline DeformHandlPlaceData deformHandlPlace;
inline SphereMesh           deformSphereMarker;

inline float               gGroupRadius      = 0.5f;
inline glm::vec3           gravity           = glm::vec3(0.0f, 0.0f, 0.0f);

// ============================================================================
// シーンスケール (registration 側と完全同期)
// ----------------------------------------------------------------------------
// 起動時に target mesh の AABB 対角線で計測し、以降のカメラ感度・距離・
// scale 依存しきい値をこの値に比例させる。registration 側と同じ運用。
//
// kRefSceneDiag = 7.36f はレジストレーション側の調整時(scale=10 時代)の
// 典型値。これを「リファレンススケール」と定義し、現在シーンの sceneDiag
// との比率 r = sceneDiag / kRefSceneDiag を全 scale 依存パラメータに掛ける
// ことで、scale=1 でも scale=10 でも同じ操作感・同じレジストレーション挙動
// を実現する。
//
// 論文化観点では kRefSceneDiag は「参考実装のキャリブレーション点」として
// 提示し、最終的には median NN distance L 等の正規化定数(Phase B)で
// 引用比に置き換える方針(registration 側と歩調を合わせる)。
// ============================================================================
inline float               gSceneDiag        = 0.0f;     // loadReferenceMeshes 後に設定
inline constexpr float     kRefSceneDiag     = 7.36f;    // registration 側と一致

// ============================================================================
// 参照表示用メッシュ(レジストレーション時の Target / Board をそのまま流用)
//   gTargetMesh = 元の長いmain.cpp の screenMesh 相当
//   gBoardMesh  = 元の長いmain.cpp の boardMesh3D 相当
// ============================================================================
inline mCutMesh*           gTargetMesh       = nullptr;
inline mCutMesh*           gBoardMesh        = nullptr;

// ============================================================================
// AutoDeform 用静的 liver メッシュ
//   元の長いmain.cpp の liverMesh3D 相当(DEFORM_MODE 入り口時点の冷凍状態)
//   AutoDeform は変形中もこの静的状態を読む
//   SoftBody の現在状態は別途 multiBody->getVisPositions(0) で渡す
// ============================================================================
inline mCutMesh*           gLiverStaticMesh  = nullptr;

// 3状態 alpha (1.0=ON / 0.5=50% / 0.0=OFF) — V/T/B キーで cycle
inline float               gOrganAlpha       = 1.0f;   // 全臓器一括 (V キー)
inline float               gTargetAlpha      = 0.5f;   // Target 表示 (T キー)
inline float               gBoardAlpha       = 0.5f;   // Board  表示 (B キー)

// 3状態 cycle ヘルパ:1.0 → 0.5 → 0.0 → 1.0 → ...
inline void cycleAlpha(float& a) {
    if      (a > 0.7f) a = 0.5f;
    else if (a > 0.2f) a = 0.0f;
    else               a = 1.0f;
}
inline const char* alphaLabel(float a) {
    if (a > 0.7f) return "ON";
    if (a > 0.2f) return "50%";
    return "OFF";
}

// ============================================================================
// AutoDeform / HEMI-AUTO 駆動グローバル
//   元の長いmain.cpp の同名グローバルと完全一致(統合時のため)
// ============================================================================
inline AutoDeform::State    gAutoDeform;
inline AutoHandleController gAutoCtrl;

inline int   gAutoDeformPresetIdx    = 0;       // 0..4 (P0 Sequential ... P4 Extreme)
inline float gMoveScale              = 0.1f;    // HEMI 1ステップで何%進めるか (-/= で増減)
inline int   gAutoDeformBoostIter    = 30;
inline float gAutoDeformBoostDamping = 0.3f;
inline bool  gAutoDeformFirstCommit  = false;

// BEFORE/AFTER スナップショット (Key 7 用)
inline std::vector<float>              gSnapBeforePositions;
inline std::vector<std::vector<float>> gSnapBeforeVisPositions;
inline std::vector<float>              gSnapAfterPositions;
inline std::vector<std::vector<float>> gSnapAfterVisPositions;
inline bool  gSnapBeforeValid = false;
inline bool  gSnapAfterValid  = false;
inline bool  gShowingAfter    = false;

// ============================================================================
// Inspect mode (Key 7 で BEFORE/AFTER 表示中に物理を完全停止する)
// ----------------------------------------------------------------------------
// rigidMode を true にすると preSolve 内の updateRigidTransform() が
// restPositionsRelative (= 起動時の rest pose) で positions を上書きしてしまうため、
// snapshot を表示しても次フレームで rest pose に戻されて画面が変わらない問題が
// 発生する。これを防ぐため、独立したフラグで preSolve/solve/postSolve/
// updateVisMeshes を全て skip する方式に切替。
//
// gInspectMode = true のとき:
//   - updateAndDraw 内の物理3点セット & updateVisMeshes は呼ばれない
//   - setPositions / setAllVisPositions で書き込んだ snapshot がそのまま保持される
//   - 描画は通常通り行われる(snapshot 状態が見える)
// ============================================================================
inline bool  gInspectMode     = false;

// Pending RMSE 表示 (settled measurement after physics relaxation)
inline float gPendingRmseImmediate = 0.0f;
inline int   gPendingRmseFrames    = 0;

// ============================================================================
// モード名(タイトルバー / デバッグ表示用)
// ============================================================================
inline const char* mainModeName(MainMode m) {
    return (m == DEFORM_MODE) ? "DEFORM" : "REGISTRATION";
}
inline const char* deformStateName(DeformHandlPlaceData::State s) {
    switch (s) {
    case DeformHandlPlaceData::RIGID_MODE:        return "RIGID";
    case DeformHandlPlaceData::HANDLE_PLACE_MODE: return "HANDLE_PLACE";
    case DeformHandlPlaceData::DEFORM_MODE:       return "DEFORM";
    case DeformHandlPlaceData::PLANECUT_MODE:     return "PLANECUT";
    }
    return "?";
}
