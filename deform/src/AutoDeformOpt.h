// AutoDeformOpt.h
//
// AutoDeform Step 6 の自動化 (Case A: stateful patience-based).
//
//   - autoStepActive    : 現在アクティブな move handle を patience model で
//                         前進方向に最適化(forward only, unwind to best)
//   - autoStepAll       : 全 move handles を round-robin で最大 gMaxPasses 周
//   - resetAllProgress  : 全 move handles の progress を 0 まで Bksp で巻き戻し
//
// 高品質オプション:
//   - gUseTwoStage = true でセット (default OFF)
//     Coarse pass (gMoveScale = gCoarseScale, default 0.25) →
//     Fine pass   (gMoveScale = gFineScale,   default 0.05)
//     Fine は coarse の best 位置から 1 coarse step 戻った地点から forward。
//     これにより coarse の grid 飛び越し領域もカバーする。
//
//   - gAutoBoostIter = 80 で auto 実行中の boost iter を上書き(default Key6 は 30)。
//     Key 6 / Bksp の runBoost が読む gAutoDeformBoostIter を一時的に書き換える
//     ことで「auto のときだけ高品質」を実現。終了時に元の値に復元。
//
// 既知の限界 (Case A 本質):
//   1. Forward only: progress 減少方向の探索はしない。
//   2. Path-dependent: 巻き戻し後の RMSE が探索中 best と微妙にずれる
//      (PBD の attached 以外 particle が velocity を保持するため)。
//      → 巻き戻し後に再計測して bestRMSE を更新するので表示と実態は一致。
//
// 将来計画 (実装スコープ外):
//   Case B : 黄金分割探索 (bidirectional line search)
//   Case C : CMA-ES (BIPOP, REG 側 CmaesRefineV3 流用)
//   並列化  : SoftBody clone + thread pool で population eval を λ 並列に
//
#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// GL ヘッダ取り込み順序衝突回避(<GLFW> がデフォで gl.h を引くため抑止)
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include "DeformGlobals.h"    // gAutoDeform, gAutoCtrl, multiBody, gTargetMesh,
                              // gMoveScale, gAutoDeformBoostIter
#include "DeformPipeline.h"   // DeformPipeline::onKey, autoMeasureRMSE, scaledMaxDist

namespace AutoDeformOpt {

// ============================================================================
// Tunables (パネル Settings expander から動的変更可)
// ============================================================================

// 単段の patience model
inline int   gMaxSteps   = 20;       // 1 handle あたり最大 step 数
inline int   gPatience   = 2;        // 連続「改善なし」許容数
inline float gEpsRatio   = 0.001f;   // 改善判定の相対閾値 (0.1%)
inline int   gMaxPasses  = 4;        // autoStepAll の最大 pass 数

// Auto 実行中の boost iter (Key 6 の手動値 gAutoDeformBoostIter=30 を一時上書き)
//   30 だと immediate RMSE と settled RMSE が ~0.1〜0.3% ずれて patience 判定
//   が不安定になる。80 だと immediate が settled にほぼ一致。
inline int   gAutoBoostIter = 80;

// 2-stage (Coarse → Fine) モード
inline bool  gUseTwoStage   = false;  // default OFF (まず単段で動作確認後 ON 推奨)
inline float gCoarseScale   = 0.25f;  // coarse step 幅 (4 grid: 0.25/0.5/0.75/1.0)
inline float gFineScale     = 0.05f;  // fine step 幅 (coarse best 近傍の精密探索)

// ============================================================================
// 結果型
// ============================================================================
enum class StopReason {
    None,
    NoHandles,        // stage<5 / numMove==0
    NoActive,         // activeMoveIdx<0
    Boundary,         // progress = 1.0 到達
    NoChange,         // stepActive が false (進めず)
    NoImprovement,    // patience 切れ
    MaxSteps,         // maxSteps 到達
    Converged,        // autoStepAll: 1 pass 全 handle で改善なし → 収束
    MaxPasses,        // autoStepAll: maxPasses 到達
};

inline const char* toStr(StopReason r) {
    switch (r) {
    case StopReason::None:          return "none";
    case StopReason::NoHandles:     return "no_handles";
    case StopReason::NoActive:      return "no_active";
    case StopReason::Boundary:      return "boundary";
    case StopReason::NoChange:      return "no_change";
    case StopReason::NoImprovement: return "no_improvement";
    case StopReason::MaxSteps:      return "max_steps";
    case StopReason::Converged:     return "converged";
    case StopReason::MaxPasses:     return "max_passes";
    }
    return "?";
}

struct SingleResult {
    int        moveIdx       = -1;
    float      startProgress = 0.0f;
    float      bestProgress  = 0.0f;
    float      startRMSE     = -1.0f;
    float      bestRMSE      = -1.0f;
    int        stepsTaken    = 0;
    int        stepsUnwound  = 0;
    bool       twoStage      = false;  // 2-stage で実行されたか
    StopReason stopReason    = StopReason::None;
};

struct AllResult {
    int        passes      = 0;
    int        totalSteps  = 0;
    float      startRMSE   = -1.0f;
    float      bestRMSE    = -1.0f;
    bool       twoStage    = false;
    StopReason stopReason  = StopReason::None;
    std::vector<SingleResult> perHandle;
};

// パネル表示用の最新結果キャッシュ
inline SingleResult gLastSingle;
inline AllResult    gLastAll;

// ============================================================================
// ヘルパ
// ============================================================================

// 現在の visMesh 状態で RMSE を測り直す。
// Key 6 / Bksp 経由で rmseLastMeasured は既に更新されているが、ループ開始時の
// baseline として「今この瞬間の RMSE」を取りたい場面で使用。
inline float measureNow() {
    if (!multiBody || !gTargetMesh) return -1.0f;
    const auto& visVerts = multiBody->getVisPositions(0);
    return DeformPipeline::autoMeasureRMSE(
        visVerts, gTargetMesh,
        /*storeAsBefore=*/false,
        DeformPipeline::scaledMaxDist(),
        /*verbose=*/false);
}

// 単一方向 (forward) の patience search コア。autoStepActive と
// 2-stage の coarse / fine の両方から呼ばれる。
//
// 引数:
//   activeIdx : 現在のアクティブ move handle インデックス
//   startProgress / startRMSE : 探索開始時点の値
//   bestProgress / bestRMSE   : in-out, 既知の best (2-stage の fine pass は
//                                coarse の best を引き継ぐ)
//   stepsTakenOut / stepsUnwoundOut : out, この pass で取った step 数
//   stopReasonOut : out, 停止理由
//
// 戻り値: なし。bestProgress / bestRMSE が更新される。物理状態は best 位置に巻き戻し済み。
inline void patienceSearchForward(
    int activeIdx,
    float startProgress, float startRMSE,
    float& bestProgress, float& bestRMSE,
    int&   stepsTakenOut,
    int&   stepsUnwoundOut,
    StopReason& stopReasonOut)
{
    int  bestStepFromStart = 0;
    int  patienceRemaining = gPatience;
    int  stepsTaken        = 0;
    bool hitBoundary       = false;
    bool noChange          = false;

    while (stepsTaken < gMaxSteps && patienceRemaining > 0) {
        const float prevProgress = gAutoCtrl.moveHandle(activeIdx).progress;
        if (prevProgress >= 1.0f - 1e-5f) {
            hitBoundary = true;
            break;
        }

        // 既存 Key 6 のロジック完全再利用
        DeformPipeline::onKey(GLFW_KEY_6, 0);

        const float newProgress = gAutoCtrl.moveHandle(activeIdx).progress;
        if (std::abs(newProgress - prevProgress) < 1e-6f) {
            noChange = true;
            break;
        }

        stepsTaken++;
        const float curRMSE = gAutoDeform.rmseLastMeasured;
        const float epsAbs  = std::max(1e-9f, bestRMSE * gEpsRatio);

        if (curRMSE < bestRMSE - epsAbs) {
            bestRMSE          = curRMSE;
            bestProgress      = newProgress;
            bestStepFromStart = stepsTaken;
            patienceRemaining = gPatience;
        } else {
            patienceRemaining--;
        }
    }

    // 停止理由
    if      (hitBoundary)            stopReasonOut = StopReason::Boundary;
    else if (noChange)                stopReasonOut = StopReason::NoChange;
    else if (patienceRemaining <= 0)  stopReasonOut = StopReason::NoImprovement;
    else                              stopReasonOut = StopReason::MaxSteps;

    // Unwind to best
    const int unwindSteps = stepsTaken - bestStepFromStart;
    for (int i = 0; i < unwindSteps; i++) {
        DeformPipeline::onKey(GLFW_KEY_BACKSPACE, 0);
    }

    stepsTakenOut   = stepsTaken;
    stepsUnwoundOut = unwindSteps;

    // Path-dependent physics: 巻き戻し後の実測値で bestRMSE 更新
    //
    // CRITICAL: best が「この pass 内」で更新された場合のみ上書きする。
    //   bestStepFromStart > 0 : この pass で best 更新 → unwind 後の物理位置 == bestProgress
    //                            → measureNow() は best 位置の現実 RMSE → 上書き OK
    //   bestStepFromStart == 0: best は前 pass からの引き継ぎ(2-stage で coarse が見つけたもの)
    //                            → unwind 後の物理位置は「この pass の start」であり
    //                              bestProgress とは別の場所。上書きすると best 情報が破壊される。
    //                            → 呼び出し側(twoStageSearch)が後で物理位置を bestProgress に戻して
    //                              再測定する責任を負う。
    if (unwindSteps > 0 && bestStepFromStart > 0) {
        const float postUnwindRMSE = measureNow();
        const float drift = std::abs(postUnwindRMSE - bestRMSE);
        const float driftRatio = (bestRMSE > 0.0f) ? (drift / bestRMSE) : 0.0f;
        if (driftRatio > 0.001f) {
            std::cout << "[AutoOpt] post-unwind drift " << driftRatio*100.0f << "%"
                      << "  recorded=" << bestRMSE
                      << "  actual="   << postUnwindRMSE << std::endl;
        }
        bestRMSE = postUnwindRMSE;
    }

    (void)startProgress;
    (void)startRMSE;
}

// Coarse → Fine 2-stage 探索 (1 handle).
//   1. gMoveScale = gCoarseScale で forward search → coarse_best
//   2. coarse_best から 1 coarse step 戻る (= coarse の "取り逃し" 領域に入る)
//   3. gMoveScale = gFineScale で forward search → fine_best
//      ※ fine search は coarse_best - gCoarseScale から forward に進むので
//        [coarse_best - gCoarseScale, coarse_best + maxFineSteps * gFineScale]
//        の連続区間をカバーする。
//   4. gMoveScale を元に戻す
//
// 注意: gMoveScale は他のコード (Key 6, Bksp) でも読まれるグローバル。
// 関数内で setter -> 終了時に必ず restore する。
inline void twoStageSearch(
    int activeIdx,
    float startProgress, float startRMSE,
    float& bestProgress, float& bestRMSE,
    int&   stepsTakenOut,
    int&   stepsUnwoundOut,
    StopReason& stopReasonOut)
{
    const float savedMoveScale = gMoveScale;

    // ===== Stage 1: Coarse =====
    gMoveScale = gCoarseScale;
    std::cout << "[AutoOpt::two-stage] >>> Coarse pass (scale=" << gCoarseScale
              << ")  start RMSE=" << bestRMSE << std::endl;

    int        cSteps = 0, cUnwound = 0;
    StopReason cReason = StopReason::None;
    patienceSearchForward(activeIdx, startProgress, startRMSE,
                          bestProgress, bestRMSE,
                          cSteps, cUnwound, cReason);

    const float coarseBest = gAutoCtrl.moveHandle(activeIdx).progress;
    std::cout << "[AutoOpt::two-stage] coarse done: steps=" << cSteps
              << " unwound=" << cUnwound
              << " best=" << bestRMSE
              << " coarseBestProgress=" << coarseBest
              << " reason=" << toStr(cReason) << std::endl;

    // ===== Stage 1.5: Step back 1 coarse step (取り逃し領域に入る) =====
    // coarseBest > 0 のとき、その下側を fine で精査するため。
    // gMoveScale が coarse のまま Bksp すれば 1 coarse 戻れる。
    int rewoundForFine = 0;
    if (coarseBest > 1e-5f) {
        DeformPipeline::onKey(GLFW_KEY_BACKSPACE, 0);
        rewoundForFine = 1;
        std::cout << "[AutoOpt::two-stage] rewound 1 coarse step to "
                  << gAutoCtrl.moveHandle(activeIdx).progress
                  << " (preparing fine pass)" << std::endl;
        // ※ ここで RMSE は更新されたが bestRMSE はそのまま (coarse best を保持)。
        //    Fine pass がここから forward に進むので、もし途中で coarse best より
        //    良い点があれば bestRMSE が更新される。
    }

    // ===== Stage 2: Fine =====
    gMoveScale = gFineScale;
    std::cout << "[AutoOpt::two-stage] >>> Fine pass (scale=" << gFineScale
              << ")  best so far=" << bestRMSE << std::endl;

    int        fSteps = 0, fUnwound = 0;
    StopReason fReason = StopReason::None;
    patienceSearchForward(activeIdx, startProgress, startRMSE,
                          bestProgress, bestRMSE,
                          fSteps, fUnwound, fReason);

    std::cout << "[AutoOpt::two-stage] fine done: steps=" << fSteps
              << " unwound=" << fUnwound
              << " best=" << bestRMSE
              << " finalProgress=" << gAutoCtrl.moveHandle(activeIdx).progress
              << " reason=" << toStr(fReason) << std::endl;

    // ===== Stage 3: Restore physical position to bestProgress =====
    //
    // patienceSearchForward は「この pass の forward 分」しか巻き戻さない。
    // Fine pass が改善を見つけられなかった場合、unwind 後の物理位置は
    // 「fine pass の start」(= coarseBest - 1 coarse step) であり、
    // bestProgress (= coarseBest) と一致しない。
    //
    // ここで gFineScale 刻みで forward / backward して bestProgress 近傍まで
    // 物理位置を戻し、最後に measureNow() で bestRMSE を上書きする。
    //
    // 注意: 物理は path-dependent なので「同じ progress に戻しても同じ RMSE」とは
    // 限らない。なので「coarse で見つけた 0.01658」と「戻った後の 0.01660」のように
    // 微妙にずれることがある。これは log に出る。
    int corrSteps = 0;
    {
        float currentProgress = gAutoCtrl.moveHandle(activeIdx).progress;
        const float tol = gFineScale * 0.6f;   // fine ステップ半分強の許容

        gMoveScale = gFineScale;  // 念のため fine に固定 (patience 中も fine だが明示)

        int safetyTraverse = 50;
        while (std::abs(currentProgress - bestProgress) > tol && safetyTraverse-- > 0) {
            if (currentProgress < bestProgress) {
                DeformPipeline::onKey(GLFW_KEY_6, 0);
            } else {
                DeformPipeline::onKey(GLFW_KEY_BACKSPACE, 0);
            }
            const float newProg = gAutoCtrl.moveHandle(activeIdx).progress;
            if (std::abs(newProg - currentProgress) < 1e-6f) break;
            currentProgress = newProg;
            corrSteps++;
        }

        if (corrSteps > 0) {
            const float restoredRMSE = measureNow();
            const float drift = std::abs(restoredRMSE - bestRMSE);
            const float driftRatio = (bestRMSE > 0.0f) ? (drift / bestRMSE) : 0.0f;
            std::cout << "[AutoOpt::two-stage] post-fine traversal: "
                      << corrSteps << " fine steps to bestProgress="
                      << bestProgress
                      << " (current=" << currentProgress << ")"
                      << "  RMSE=" << restoredRMSE
                      << " (was recorded " << bestRMSE
                      << ", drift=" << driftRatio*100.0f << "%)" << std::endl;
            bestRMSE = restoredRMSE;
        }
    }

    // ===== Restore gMoveScale =====
    gMoveScale = savedMoveScale;

    // Aggregate
    stepsTakenOut   = cSteps + rewoundForFine + fSteps + corrSteps;
    stepsUnwoundOut = cUnwound + rewoundForFine + fUnwound;
    // 停止理由は fine pass のものを採用 (最後に動いた pass)
    stopReasonOut   = fReason;
}

// ============================================================================
// Public API: autoStepActive (single handle)
// ============================================================================
inline SingleResult autoStepActive() {
    SingleResult r;

    if (gAutoDeform.stage < 5 || gAutoCtrl.numMove() == 0) {
        r.stopReason = StopReason::NoHandles;
        std::cout << "[AutoOpt::single] stage<5 or no move handles; "
                     "run Stage 5 first." << std::endl;
        gLastSingle = r;
        return r;
    }
    const int activeIdx = gAutoCtrl.activeMoveIdx();
    if (activeIdx < 0) {
        r.stopReason = StopReason::NoActive;
        std::cout << "[AutoOpt::single] no active handle. press Next first."
                  << std::endl;
        gLastSingle = r;
        return r;
    }

    r.moveIdx       = activeIdx;
    r.startProgress = gAutoCtrl.moveHandle(activeIdx).progress;
    r.startRMSE     = measureNow();
    r.bestProgress  = r.startProgress;
    r.bestRMSE      = r.startRMSE;
    r.twoStage      = gUseTwoStage;

    // Auto 中の boost iter 上書き (immediate ≈ settled に近づける)
    const int savedBoost = gAutoDeformBoostIter;
    gAutoDeformBoostIter = gAutoBoostIter;

    std::cout << "\n[AutoOpt::single] === start move=" << activeIdx
              << "  startProgress=" << r.startProgress
              << "  startRMSE=" << r.startRMSE
              << "  (mode=" << (gUseTwoStage ? "2-stage" : "1-stage")
              << " maxSteps=" << gMaxSteps
              << " patience=" << gPatience
              << " eps=" << gEpsRatio
              << " boost=" << gAutoBoostIter
              << ")" << std::endl;

    if (gUseTwoStage) {
        twoStageSearch(activeIdx, r.startProgress, r.startRMSE,
                       r.bestProgress, r.bestRMSE,
                       r.stepsTaken, r.stepsUnwound, r.stopReason);
    } else {
        patienceSearchForward(activeIdx, r.startProgress, r.startRMSE,
                              r.bestProgress, r.bestRMSE,
                              r.stepsTaken, r.stepsUnwound, r.stopReason);
    }

    // restore boost iter
    gAutoDeformBoostIter = savedBoost;

    const float improvement = r.startRMSE - r.bestRMSE;
    std::cout << "[AutoOpt::single] done move=" << activeIdx
              << "  totalSteps=" << r.stepsTaken
              << "  unwound=" << r.stepsUnwound
              << "  RMSE " << r.startRMSE << " -> " << r.bestRMSE
              << "  (" << (improvement >= 0 ? "+" : "") << improvement << ")"
              << "  bestProgress=" << r.bestProgress
              << "  reason=" << toStr(r.stopReason) << std::endl;

    gLastSingle = r;
    return r;
}

// ============================================================================
// Public API: autoStepAll (round-robin all handles)
// ============================================================================
inline AllResult autoStepAll() {
    AllResult R;

    if (gAutoDeform.stage < 5 || gAutoCtrl.numMove() == 0) {
        R.stopReason = StopReason::NoHandles;
        std::cout << "[AutoOpt::all] stage<5 or no move handles; "
                     "run Stage 5 first." << std::endl;
        gLastAll = R;
        return R;
    }

    const int N = gAutoCtrl.numMove();
    R.startRMSE = measureNow();
    R.bestRMSE  = R.startRMSE;
    R.twoStage  = gUseTwoStage;

    std::cout << "\n[AutoOpt::all] === start numMove=" << N
              << "  startRMSE=" << R.startRMSE
              << "  (mode=" << (gUseTwoStage ? "2-stage" : "1-stage")
              << " maxPasses=" << gMaxPasses << ")" << std::endl;

    const int savedActive = std::max(0, gAutoCtrl.activeMoveIdx());

    for (int pass = 0; pass < gMaxPasses; pass++) {
        bool anyImprovedInPass = false;
        R.passes++;
        std::cout << "[AutoOpt::all] --- pass " << (pass + 1) << " / "
                  << gMaxPasses << " ---" << std::endl;

        for (int h = 0; h < N; h++) {
            gAutoCtrl.setActiveMove(h);
            SingleResult sr = autoStepActive();
            R.perHandle.push_back(sr);
            R.totalSteps += sr.stepsTaken;

            const float epsAbs = std::max(1e-9f, R.bestRMSE * gEpsRatio);
            if (sr.bestRMSE >= 0.0f && sr.bestRMSE < R.bestRMSE - epsAbs) {
                R.bestRMSE = sr.bestRMSE;
                anyImprovedInPass = true;
            }
        }

        if (!anyImprovedInPass) {
            R.stopReason = StopReason::Converged;
            break;
        }
    }
    if (R.stopReason == StopReason::None) {
        R.stopReason = StopReason::MaxPasses;
    }

    gAutoCtrl.setActiveMove(savedActive);

    const float improvement = R.startRMSE - R.bestRMSE;
    std::cout << "[AutoOpt::all] === done"
              << "  passes=" << R.passes
              << "  totalSteps=" << R.totalSteps
              << "  RMSE " << R.startRMSE << " -> " << R.bestRMSE
              << "  (" << (improvement >= 0 ? "+" : "") << improvement << ")"
              << "  reason=" << toStr(R.stopReason) << std::endl;

    gLastAll = R;
    return R;
}

// ============================================================================
// Public API: resetAllProgress (全 move handle progress=0 まで Bksp 連打)
// ============================================================================
//
// Auto を試した後「やり直したい」用。
// safety リミットあり (1 handle あたり最大 200 step) — gMoveScale が極端に
// 小さいケースでも無限ループにならないように。
inline void resetAllProgress() {
    if (!multiBody || gAutoCtrl.numMove() == 0) {
        std::cout << "[AutoOpt::reset] no handles to reset" << std::endl;
        return;
    }
    const int N = gAutoCtrl.numMove();
    const int savedActive = std::max(0, gAutoCtrl.activeMoveIdx());

    std::cout << "\n[AutoOpt::reset] resetting " << N
              << " move handles to progress=0 ..." << std::endl;

    int totalBacksteps = 0;
    for (int h = 0; h < N; h++) {
        gAutoCtrl.setActiveMove(h);
        int safety = 200;
        int taps   = 0;
        while (gAutoCtrl.moveHandle(h).progress > 1e-5f && safety-- > 0) {
            DeformPipeline::onKey(GLFW_KEY_BACKSPACE, 0);
            taps++;
        }
        totalBacksteps += taps;
        std::cout << "  move=" << h
                  << "  backsteps=" << taps
                  << "  final progress=" << gAutoCtrl.moveHandle(h).progress
                  << std::endl;
    }

    gAutoCtrl.setActiveMove(savedActive);

    std::cout << "[AutoOpt::reset] done. total backsteps=" << totalBacksteps
              << std::endl;
}

}  // namespace AutoDeformOpt
