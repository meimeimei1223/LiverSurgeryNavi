#ifndef HANDLE_CONTROLLER_BASE_H
#define HANDLE_CONTROLLER_BASE_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <glm/glm.hpp>
#include "SoftBody.h"

class HandleControllerBase {
public:
    struct Handle {
        int                    centerVertex = -1;
        std::vector<int>       vertices;
        std::vector<glm::vec3> relativePositions;
        glm::vec3              initialCenter = glm::vec3(0.0f);
        glm::vec3              dirVec        = glm::vec3(0.0f);
        float                  radius        = 0.5f;
        bool                   isFixed       = false;
        float                  progress      = 0.0f;
        // Fine-tune モードでマウスドラッグした手動補正量。
        //   center = initialCenter + dirVec * progress + manualOffset
        // AUTO の progress とは独立。Reset all progress では消えず、
        // 専用の clearManualOffsets() でのみ 0 に戻る。
        glm::vec3              manualOffset  = glm::vec3(0.0f);
    };

    enum HandleKind { KIND_FIX = 0, KIND_MOVE = 1 };

    virtual ~HandleControllerBase() = default;

    void clear(SoftBody* body) {
        if (!body) {
            fixHandles_.clear();
            moveHandles_.clear();
            savedInvMasses_.clear();
            savedIdx_.clear();
            activeMoveIdx_ = -1;
            return;
        }
        for (size_t i = 0; i < savedIdx_.size(); i++) {
            body->setInvMass(savedIdx_[i], savedInvMasses_[i]);
        }
        savedInvMasses_.clear();
        savedIdx_.clear();
        fixHandles_.clear();
        moveHandles_.clear();
        activeMoveIdx_ = -1;
    }

    int numFix()  const { return static_cast<int>(fixHandles_.size()); }
    int numMove() const { return static_cast<int>(moveHandles_.size()); }
    int activeMoveIdx() const { return activeMoveIdx_; }

    const Handle& fixHandle(int i) const { return fixHandles_[i]; }
    const Handle& moveHandle(int i) const { return moveHandles_[i]; }

    void selectNextMove() {
        if (moveHandles_.empty()) {
            activeMoveIdx_ = -1;
            return;
        }
        activeMoveIdx_ = (activeMoveIdx_ < 0)
            ? 0
            : (activeMoveIdx_ + 1) % static_cast<int>(moveHandles_.size());
        std::cout << "[HandleCtrl] active move=" << activeMoveIdx_
                  << " progress=" << moveHandles_[activeMoveIdx_].progress << std::endl;
    }

    void setActiveMove(int idx) {
        if (idx < 0 || idx >= (int)moveHandles_.size()) return;
        activeMoveIdx_ = idx;
    }

    bool stepActive(SoftBody* body, float delta) {
        if (!body || activeMoveIdx_ < 0 || activeMoveIdx_ >= (int)moveHandles_.size()) return false;
        Handle& h = moveHandles_[activeMoveIdx_];
        float prev = h.progress;
        float newProg = prev + delta;
        float steps = std::round(newProg / std::max(1e-6f, std::abs(delta)));
        newProg = steps * std::abs(delta);
        if (delta < 0.0f && newProg < 1e-5f) newProg = 0.0f;
        newProg = std::max(0.0f, std::min(1.0f, newProg));
        if (std::abs(newProg - prev) < 1e-6f) return false;
        h.progress = newProg;
        applyProgress(body, activeMoveIdx_);
        return true;
    }

    void applyProgress(SoftBody* body, int moveIdx) {
        if (!body) return;
        if (moveIdx < 0 || moveIdx >= (int)moveHandles_.size()) return;
        Handle& h = moveHandles_[moveIdx];
        // AUTO の 1 次元拘束 (dirVec * progress) に、Fine-tune の手動補正
        // (manualOffset) を加えた位置を center とする。
        glm::vec3 newCenter = h.initialCenter + h.dirVec * h.progress + h.manualOffset;
        for (size_t i = 0; i < h.vertices.size(); i++) {
            int vi = h.vertices[i];
            glm::vec3 newPos = newCenter + h.relativePositions[i];
            body->setParticlePosition(vi, newPos);
            body->setParticlePrevPosition(vi, newPos);
            body->setParticleVelocity(vi, glm::vec3(0.0f));
        }
    }

    // fix handle 用の位置適用。fix は progress/dirVec を持たないが、Fine-tune の
    // manualOffset は move と同じく効かせたい (通常グラブが種類を区別しないのと同様、
    // Fine-tune も fix/move を統一的に扱う)。center = initialCenter + manualOffset。
    void applyFixHandle(SoftBody* body, int fixIdx) {
        if (!body) return;
        if (fixIdx < 0 || fixIdx >= (int)fixHandles_.size()) return;
        Handle& h = fixHandles_[fixIdx];
        glm::vec3 newCenter = h.initialCenter + h.manualOffset;
        for (size_t i = 0; i < h.vertices.size(); i++) {
            int vi = h.vertices[i];
            glm::vec3 newPos = newCenter + h.relativePositions[i];
            body->setParticlePosition(vi, newPos);
            body->setParticlePrevPosition(vi, newPos);
            body->setParticleVelocity(vi, glm::vec3(0.0f));
        }
    }

    void reapplyAllProgress(SoftBody* body) {
        if (!body) return;
        // fix も manualOffset 込みで再適用 (applyFixHandle 経由)。
        for (int i = 0; i < (int)fixHandles_.size(); i++) {
            applyFixHandle(body, i);
        }
        for (int i = 0; i < (int)moveHandles_.size(); i++) {
            applyProgress(body, i);
        }
    }

    void runBoost(SoftBody* body, int iter, float damping) {
        if (!body) return;
        float dampSaved = body->getDamping();
        body->setDamping(damping);
        float stepDt = 1.0f / 60.0f;
        glm::vec3 noGravity(0.0f);
        for (int k = 0; k < iter; k++) {
            body->preSolve(stepDt, noGravity);
            body->solve(stepDt);
            body->postSolve(stepDt);
        }
        body->setDamping(dampSaved);
        body->updateVisMeshes();
    }

    // ========================================================================
    // Fine-tune モード支援 (AUTO 後の手動微調整)
    //   move handle を「球を直接クリックして掴み、カメラ平面内でドラッグ」する
    //   ために必要な API。center 計算と同じ式で現在位置を返す / マウス補正を
    //   manualOffset に積む / 手動分だけクリアする。
    // ========================================================================

    // move handle の現在の world center (AUTO + 手動補正込み)。
    glm::vec3 moveHandleCenter(int moveIdx) const {
        if (moveIdx < 0 || moveIdx >= (int)moveHandles_.size()) return glm::vec3(0.0f);
        const Handle& h = moveHandles_[moveIdx];
        return h.initialCenter + h.dirVec * h.progress + h.manualOffset;
    }

    // world 座標 p に最も近い move handle を返す (半径内のみ)。
    //   見つからなければ -1。Fine-tune の球クリック判定に使う。
    //   radiusScale: handle.radius にこの倍率を掛けた距離を当たり半径とする
    //   (描画スフィアが radius*0.5 程度なので、掴みやすさのため少し大きめ可)。
    int pickMoveHandle(const glm::vec3& p, float radiusScale = 1.0f) const {
        int   best   = -1;
        float bestD2 = 1e30f;
        for (int i = 0; i < (int)moveHandles_.size(); i++) {
            glm::vec3 c = moveHandleCenter(i);
            float d2 = glm::dot(p - c, p - c);
            float r  = moveHandles_[i].radius * radiusScale;
            if (d2 <= r * r && d2 < bestD2) {
                bestD2 = d2;
                best   = i;
            }
        }
        return best;
    }

    // move handle の manualOffset を絶対値で設定し、body に即反映。
    //   Fine-tune ドラッグ中に毎フレーム呼ぶ。物理 solve は呼び出し側が回す。
    void setManualOffset(SoftBody* body, int moveIdx, const glm::vec3& offset) {
        if (moveIdx < 0 || moveIdx >= (int)moveHandles_.size()) return;
        moveHandles_[moveIdx].manualOffset = offset;
        applyProgress(body, moveIdx);
    }

    glm::vec3 manualOffset(int moveIdx) const {
        if (moveIdx < 0 || moveIdx >= (int)moveHandles_.size()) return glm::vec3(0.0f);
        return moveHandles_[moveIdx].manualOffset;
    }

    // ------------------------------------------------------------------------
    // 統一ハンドルアクセス (fix + move を 1 本の通し番号で扱う)
    //   Fine-tune は fix/move を区別しないので、main.cpp のヒットテスト/ドラッグを
    //   1 ループで書けるよう、通し番号 uIdx でアクセスする API を用意する。
    //     uIdx = 0 .. numFix-1            : fix handle   (uIdx)
    //     uIdx = numFix .. numFix+numMove : move handle  (uIdx - numFix)
    //   どちらも manualOffset を持ち、setUnifiedManualOffset で位置が即反映される。
    // ------------------------------------------------------------------------
    int numHandles() const { return numFix() + numMove(); }

    bool isUnifiedFix(int uIdx) const { return uIdx >= 0 && uIdx < numFix(); }

    // 通し番号 uIdx のハンドル現在 center (manualOffset 込み)。
    glm::vec3 unifiedHandleCenter(int uIdx) const {
        if (uIdx < 0 || uIdx >= numHandles()) return glm::vec3(0.0f);
        if (isUnifiedFix(uIdx)) {
            const Handle& h = fixHandles_[uIdx];
            return h.initialCenter + h.manualOffset;          // fix は progress 無し
        }
        return moveHandleCenter(uIdx - numFix());             // move は progress 込み
    }

    float unifiedHandleRadius(int uIdx) const {
        if (uIdx < 0 || uIdx >= numHandles()) return 0.0f;
        return isUnifiedFix(uIdx) ? fixHandles_[uIdx].radius
                                  : moveHandles_[uIdx - numFix()].radius;
    }

    glm::vec3 unifiedManualOffset(int uIdx) const {
        if (uIdx < 0 || uIdx >= numHandles()) return glm::vec3(0.0f);
        return isUnifiedFix(uIdx) ? fixHandles_[uIdx].manualOffset
                                  : moveHandles_[uIdx - numFix()].manualOffset;
    }

    // 通し番号 uIdx の manualOffset を設定し、body に即反映。
    void setUnifiedManualOffset(SoftBody* body, int uIdx, const glm::vec3& offset) {
        if (uIdx < 0 || uIdx >= numHandles()) return;
        if (isUnifiedFix(uIdx)) {
            fixHandles_[uIdx].manualOffset = offset;
            applyFixHandle(body, uIdx);
        } else {
            int m = uIdx - numFix();
            moveHandles_[m].manualOffset = offset;
            applyProgress(body, m);
        }
    }

    // 全 fix+move handle の手動補正を 0 に戻す (AUTO progress は保持)。
    void clearManualOffsets(SoftBody* body) {
        int cleared = 0;
        for (int i = 0; i < (int)fixHandles_.size(); i++) {
            if (glm::dot(fixHandles_[i].manualOffset, fixHandles_[i].manualOffset) > 1e-12f)
                cleared++;
            fixHandles_[i].manualOffset = glm::vec3(0.0f);
            if (body) applyFixHandle(body, i);
        }
        for (int i = 0; i < (int)moveHandles_.size(); i++) {
            if (glm::dot(moveHandles_[i].manualOffset, moveHandles_[i].manualOffset) > 1e-12f)
                cleared++;
            moveHandles_[i].manualOffset = glm::vec3(0.0f);
            if (body) applyProgress(body, i);
        }
        std::cout << "[HandleCtrl] cleared manual offsets on " << cleared
                  << " handles (fix+move, progress preserved)" << std::endl;
    }

    bool hasAnyManualOffset() const {
        for (const auto& h : fixHandles_) {
            if (glm::dot(h.manualOffset, h.manualOffset) > 1e-12f) return true;
        }
        for (const auto& h : moveHandles_) {
            if (glm::dot(h.manualOffset, h.manualOffset) > 1e-12f) return true;
        }
        return false;
    }

protected:
    std::vector<Handle> fixHandles_;
    std::vector<Handle> moveHandles_;
    int activeMoveIdx_ = -1;

    std::vector<int>   savedIdx_;
    std::vector<float> savedInvMasses_;

    bool registerHandle_(
        SoftBody* body,
        const glm::vec3& centerWorld,
        const glm::vec3& dirVec,
        float radius,
        bool  isFixed,
        Handle& out)
    {
        if (!body) return false;
        size_t N = body->getNumParticles();
        float minD2 = 1e30f;
        int centerIdx = -1;
        for (size_t i = 0; i < N; i++) {
            glm::vec3 p = body->getParticlePosition(static_cast<int>(i));
            float d2 = glm::dot(p - centerWorld, p - centerWorld);
            if (d2 < minD2) { minD2 = d2; centerIdx = static_cast<int>(i); }
        }
        if (centerIdx < 0) return false;

        glm::vec3 snappedCenter = body->getParticlePosition(centerIdx);
        out.centerVertex  = centerIdx;
        out.initialCenter = snappedCenter;
        out.dirVec        = dirVec;
        out.radius        = radius;
        out.isFixed       = isFixed;
        out.progress      = 0.0f;
        out.vertices.clear();
        out.relativePositions.clear();

        float r2 = radius * radius;
        for (size_t i = 0; i < N; i++) {
            int idx = static_cast<int>(i);
            glm::vec3 p = body->getParticlePosition(idx);
            glm::vec3 rel = p - snappedCenter;
            if (glm::dot(rel, rel) > r2) continue;
            out.vertices.push_back(idx);
            out.relativePositions.push_back(rel);
            bool alreadySaved = false;
            for (int s : savedIdx_) { if (s == idx) { alreadySaved = true; break; } }
            if (!alreadySaved) {
                savedIdx_.push_back(idx);
                savedInvMasses_.push_back(body->getInvMass(idx));
            }
            body->setInvMass(idx, 0.0f);
            body->setParticleVelocity(idx, glm::vec3(0.0f));
            body->setParticlePrevPosition(idx, p);
        }
        return true;
    }
};

#endif
