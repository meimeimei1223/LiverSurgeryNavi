#ifndef SEQUENTIAL_DEFORM_CONTROLLER_H
#define SEQUENTIAL_DEFORM_CONTROLLER_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <glm/glm.hpp>
#include "AutoDeform.h"
#include "SoftBody.h"
#include "mCutMesh.h"

class SequentialDeformController {
public:
    enum SlotState {
        SLOT_UNTOUCHED = 0,
        SLOT_EDITED    = 1,
        SLOT_SKIPPED   = 2
    };

    struct SlotInfo {
        SlotState state    = SLOT_UNTOUCHED;
        float     progress = 0.0f;
    };

    SequentialDeformController() = default;

    void initialize(
        AutoDeform::State& st,
        const std::vector<float>& origVisVerts)
    {
        AutoDeform::initSequentialQueue(st, origVisVerts);
        int N = static_cast<int>(st.seqMoveQueue.size());
        slotInfo_.assign(N, SlotInfo{});
        activeSlot_ = -1;
        st.seqAppliedCount = 0;
    }

    void resetAll() {
        for (auto& s : slotInfo_) {
            s.state    = SLOT_UNTOUCHED;
            s.progress = 0.0f;
        }
        activeSlot_ = -1;
    }

    int size() const { return static_cast<int>(slotInfo_.size()); }
    int activeSlot() const { return activeSlot_; }
    float activeProgress() const {
        if (activeSlot_ < 0 || activeSlot_ >= (int)slotInfo_.size()) return 0.0f;
        return slotInfo_[activeSlot_].progress;
    }
    SlotState slotState(int slot) const {
        if (slot < 0 || slot >= (int)slotInfo_.size()) return SLOT_UNTOUCHED;
        return slotInfo_[slot].state;
    }
    float slotProgress(int slot) const {
        if (slot < 0 || slot >= (int)slotInfo_.size()) return 0.0f;
        return slotInfo_[slot].progress;
    }

    bool isSelectableByUser(int slot) const {
        if (slot < 0 || slot >= (int)slotInfo_.size()) return false;
        return slotInfo_[slot].state != SLOT_SKIPPED;
    }

    bool selectFirst(
        AutoDeform::State& st,
        const std::vector<float>& curVisVerts,
        float percentileLo,
        float percentileHi,
        float minSeparation,
        float rMove)
    {
        int found = findNextSlot(st, -1, percentileLo, percentileHi,
                                 minSeparation, curVisVerts, /*forward=*/true);
        if (found < 0) return false;
        return setActive(st, found, curVisVerts, rMove);
    }

    bool selectNext(
        AutoDeform::State& st,
        const std::vector<float>& curVisVerts,
        float percentileLo,
        float percentileHi,
        float minSeparation,
        float rMove)
    {
        int from = activeSlot_;
        int found = findNextSlot(st, from, percentileLo, percentileHi,
                                 minSeparation, curVisVerts, /*forward=*/true);
        if (found < 0) return false;
        return setActive(st, found, curVisVerts, rMove);
    }

    bool commit(
        AutoDeform::State& st,
        SoftBody* body,
        mCutMesh* screenMesh,
        int gridW, int gridH, float depthScale,
        float moveScale,
        float rMove,
        const AutoDeform::Preset& preset,
        int boostIter, float boostDamping,
        float& outImmediateRMSE)
    {
        outImmediateRMSE = -1.0f;
        if (!body || activeSlot_ < 0) return false;

        if (!firstCommitDone_) {
            body->setRigidMode(true);
            body->fullReset();
            st.seqPinnedParticleIdx.clear();
            st.seqPinnedTargets.clear();
            int nFix = 0;
            for (const auto& h : st.fixHandles) {
                nFix += AutoDeform::pinParticlesFromHandle(st, body, h);
            }
            std::cout << "[Seq] first commit: fix pinned=" << nFix << std::endl;
            firstCommitDone_ = true;
        }

        int resolvedThrSrc = (st.statsInterior.median > 0.0f) ? 1 : 0;
        float resolveThr = resolvedThrSrc ? st.statsInterior.median : 0.05f;
        int consumed = AutoDeform::autoConsumeResolvedCorrespondences(
            st, body->getVisPositions(0), resolveThr);
        if (consumed > 0) {
            for (int k = 0; k < (int)st.seqMoveUsed.size() && k < (int)slotInfo_.size(); k++) {
                if (st.seqMoveUsed[k] && slotInfo_[k].state != SLOT_EDITED) {
                    slotInfo_[k].state = SLOT_SKIPPED;
                }
            }
        }

        if (lastCommittedSlot_ >= 0 && lastCommittedSlot_ != activeSlot_) {
            AutoDeform::pinParticlesFromHandle(st, body, lastCommittedHandle_);
        }

        SlotInfo& cur = slotInfo_[activeSlot_];
        float prevProgress = cur.progress;
        float newProg = cur.progress + moveScale;
        float steps = std::round(newProg / moveScale);
        newProg = std::min(1.0f, steps * moveScale);
        if (newProg <= prevProgress + 1e-6f && prevProgress >= 1.0f - 1e-6f) {
            std::cout << "[6] slot=" << activeSlot_ << " already at goal (progress=1.0), no-op" << std::endl;
            return false;
        }
        float prevRMSE = st.rmseLastMeasured;
        float baseRMSE = st.rmseBeforeDeform;
        cur.progress = newProg;
        cur.state = (cur.progress > 1e-5f) ? SLOT_EDITED : SLOT_UNTOUCHED;
        st.seqMoveUsed[activeSlot_] = (cur.state == SLOT_EDITED);

        int corrIdx = st.seqMoveQueue[activeSlot_];
        AutoDeform::Handle h;
        if (!AutoDeform::buildHandleFromCorrespondence(
                st, corrIdx, body->getVisPositions(0), rMove, h, cur.progress)) {
            return false;
        }
        st.seqLastMove      = h;
        st.seqHasLastMove   = true;
        st.seqLastQueueSlot = activeSlot_;
        st.seqLastCorrIdx   = corrIdx;
        st.seqAttachedMove    = h;
        st.seqHasAttachedMove = true;
        st.moveHandles.clear();
        st.moveHandles.push_back(h);
        st.seqAppliedCount++;

        AutoDeform::applySequential(st, body, &h, preset.fixCompliance, preset.moveCompliance);

        body->setRigidMode(false);
        runBoost(body, boostIter, boostDamping);

        lastCommittedSlot_   = activeSlot_;
        lastCommittedHandle_ = h;

        outImmediateRMSE = AutoDeform::measureRMSE(
            st, body->getVisPositions(0), screenMesh,
            gridW, gridH, depthScale,
            /*storeAsBefore=*/false, 1.0f, /*verbose=*/false);

        float stepDelta = (prevRMSE > 0.0f) ? (outImmediateRMSE - prevRMSE) : 0.0f;
        float baseDelta = (baseRMSE > 0.0f) ? (outImmediateRMSE - baseRMSE) : 0.0f;
        std::cout << "[6] slot=" << activeSlot_ << " corr=" << corrIdx
                  << " progress " << prevProgress << " -> " << cur.progress
                  << "  RMSE=" << outImmediateRMSE
                  << "  step:" << (stepDelta>=0?"+":"") << stepDelta << (stepDelta<0?" DOWN":" UP")
                  << "  base:" << (baseDelta>=0?"+":"") << baseDelta << (baseDelta<0?" DOWN":" UP")
                  << std::endl;
        return true;
    }

    bool undo(
        AutoDeform::State& st,
        SoftBody* body,
        mCutMesh* screenMesh,
        int gridW, int gridH, float depthScale,
        float moveScale,
        float rMove,
        const AutoDeform::Preset& preset,
        int boostIter, float boostDamping,
        float& outImmediateRMSE)
    {
        outImmediateRMSE = -1.0f;
        if (!body || activeSlot_ < 0) return false;
        if (!st.seqHasAttachedMove) return false;

        SlotInfo& cur = slotInfo_[activeSlot_];
        if (cur.progress <= 1e-5f) {
            std::cout << "[Bksp] slot=" << activeSlot_ << " already at origin, no-op" << std::endl;
            return false;
        }
        float prev = cur.progress;
        float newProg = cur.progress - moveScale;
        float steps = std::round(newProg / moveScale);
        newProg = std::max(0.0f, steps * moveScale);
        if (newProg < 1e-5f) newProg = 0.0f;
        float prevRMSE = st.rmseLastMeasured;
        float baseRMSE = st.rmseBeforeDeform;
        cur.progress = newProg;
        if (cur.progress <= 1e-5f) {
            cur.state = SLOT_UNTOUCHED;
            st.seqMoveUsed[activeSlot_] = false;
        }

        int corrIdx = st.seqMoveQueue[activeSlot_];
        AutoDeform::Handle h;
        if (!AutoDeform::buildHandleFromCorrespondence(
                st, corrIdx, body->getVisPositions(0), rMove, h, cur.progress)) {
            return false;
        }
        st.seqLastMove      = h;
        st.seqHasLastMove   = true;
        st.seqLastCorrIdx   = corrIdx;
        st.seqLastQueueSlot = activeSlot_;
        st.seqAttachedMove    = h;
        st.seqHasAttachedMove = true;
        st.moveHandles.clear();
        st.moveHandles.push_back(h);

        AutoDeform::applySequential(st, body, &h, preset.fixCompliance, preset.moveCompliance);

        body->setRigidMode(false);
        runBoost(body, boostIter, boostDamping);

        lastCommittedSlot_   = activeSlot_;
        lastCommittedHandle_ = h;

        outImmediateRMSE = AutoDeform::measureRMSE(
            st, body->getVisPositions(0), screenMesh,
            gridW, gridH, depthScale,
            /*storeAsBefore=*/false, 1.0f, /*verbose=*/false);

        float stepDelta = (prevRMSE > 0.0f) ? (outImmediateRMSE - prevRMSE) : 0.0f;
        float baseDelta = (baseRMSE > 0.0f) ? (outImmediateRMSE - baseRMSE) : 0.0f;
        std::cout << "[Bksp] slot=" << activeSlot_ << " corr=" << corrIdx
                  << " progress " << prev << " -> " << cur.progress
                  << (cur.progress <= 0.0f ? "  [AT ORIGIN]" : "")
                  << "  RMSE=" << outImmediateRMSE
                  << "  step:" << (stepDelta>=0?"+":"") << stepDelta << (stepDelta<0?" DOWN":" UP")
                  << "  base:" << (baseDelta>=0?"+":"") << baseDelta << (baseDelta<0?" DOWN":" UP")
                  << std::endl;
        return true;
    }

    bool firstCommitDone() const { return firstCommitDone_; }
    void resetCommitFlag() { firstCommitDone_ = false; lastCommittedSlot_ = -1; }

private:
    std::vector<SlotInfo>  slotInfo_;
    int                    activeSlot_        = -1;
    bool                   firstCommitDone_   = false;
    int                    lastCommittedSlot_ = -1;
    AutoDeform::Handle     lastCommittedHandle_;

    bool setActive(
        AutoDeform::State& st,
        int slot,
        const std::vector<float>& curVisVerts,
        float rMove)
    {
        if (slot < 0 || slot >= (int)slotInfo_.size()) return false;
        activeSlot_ = slot;
        int corrIdx = st.seqMoveQueue[slot];
        float prog = slotInfo_[slot].progress;
        AutoDeform::Handle h;
        if (!AutoDeform::buildHandleFromCorrespondence(
                st, corrIdx, curVisVerts, rMove, h, prog)) return false;
        st.seqLastMove      = h;
        st.seqHasLastMove   = true;
        st.seqLastQueueSlot = slot;
        st.seqLastCorrIdx   = corrIdx;
        st.moveHandles.clear();
        st.moveHandles.push_back(h);
        std::cout << "[A] slot=" << slot << " corr=" << corrIdx
                  << " state=" << (int)slotInfo_[slot].state
                  << " progress=" << prog << std::endl;
        return true;
    }

    int findNextSlot(
        const AutoDeform::State& st,
        int fromSlot,
        float percentileLo, float percentileHi,
        float minSeparation,
        const std::vector<float>& curVisVerts,
        bool forward) const
    {
        int N = static_cast<int>(st.seqMoveQueue.size());
        if (N <= 0) return -1;
        int loIdx = std::max(0, static_cast<int>(std::floor(percentileLo * N)));
        int hiIdx = std::min(N - 1, static_cast<int>(std::floor(percentileHi * N)));
        if (loIdx > hiIdx) return -1;

        auto getCenter = [&](int corrIdx) -> glm::vec3 {
            if (corrIdx < 0) return glm::vec3(0);
            if (corrIdx < (int)st.seqVisIdxOf.size()) {
                int v = st.seqVisIdxOf[corrIdx];
                if (v >= 0 && (size_t)(v*3+2) < curVisVerts.size()) {
                    return glm::vec3(curVisVerts[v*3], curVisVerts[v*3+1], curVisVerts[v*3+2]);
                }
            }
            return st.correspondences[corrIdx].srcPos;
        };

        std::vector<glm::vec3> skipPositions;
        if (minSeparation > 0.0f) {
            skipPositions.reserve(N);
            for (int i = 0; i < N; i++) {
                if (slotInfo_[i].state == SLOT_SKIPPED)
                    skipPositions.push_back(getCenter(st.seqMoveQueue[i]));
            }
        }
        float minSep2 = minSeparation * minSeparation;

        auto canPick = [&](int i) -> bool {
            if (i == fromSlot) return false;
            if (slotInfo_[i].state == SLOT_SKIPPED) return false;
            if (minSeparation > 0.0f) {
                glm::vec3 c = getCenter(st.seqMoveQueue[i]);
                for (const auto& p : skipPositions) {
                    glm::vec3 dv = c - p;
                    if (glm::dot(dv, dv) < minSep2) return false;
                }
            }
            return true;
        };

        int start = (fromSlot < loIdx) ? loIdx : (fromSlot + (forward ? 1 : -1));
        if (forward) {
            for (int i = std::max(loIdx, start); i <= hiIdx; i++)
                if (canPick(i)) return i;
            for (int i = loIdx; i < std::max(loIdx, start); i++)
                if (canPick(i)) return i;
        } else {
            for (int i = std::min(hiIdx, start); i >= loIdx; i--)
                if (canPick(i)) return i;
            for (int i = hiIdx; i > std::min(hiIdx, start); i--)
                if (canPick(i)) return i;
        }
        return -1;
    }

    void runBoost(SoftBody* body, int iter, float damping) {
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
};

inline void drawSequentialCandidatesController(
    const AutoDeform::State& st,
    const SequentialDeformController& ctrl,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    const std::vector<float>& curVisVerts,
    float percentileLo,
    float percentileHi,
    float minSeparation,
    float radius)
{
    if (st.seqMoveQueue.empty()) return;

    int N = static_cast<int>(st.seqMoveQueue.size());
    int loIdx = std::max(0, static_cast<int>(std::floor(percentileLo * N)));
    int hiIdx = std::min(N - 1, static_cast<int>(std::floor(percentileHi * N)));

    static const glm::vec3 colSkipped  (0.35f, 0.35f, 0.35f);
    static const glm::vec3 colSkipTgt  (0.20f, 0.20f, 0.20f);
    static const glm::vec3 colEdited   (0.10f, 0.85f, 0.30f);
    static const glm::vec3 colEditTgt  (0.05f, 0.40f, 0.15f);
    static const glm::vec3 colCurrent  (1.00f, 0.55f, 0.00f);
    static const glm::vec3 colCurTgt   (1.00f, 0.10f, 0.10f);
    static const glm::vec3 colPlanned  (1.00f, 1.00f, 0.10f);
    static const glm::vec3 colPlanTgt  (0.10f, 0.50f, 1.00f);

    auto getCenter = [&](int corrIdx) -> glm::vec3 {
        if (corrIdx < 0 || corrIdx >= (int)st.correspondences.size())
            return glm::vec3(0);
        if (corrIdx < (int)st.seqVisIdxOf.size()) {
            int vIdx = st.seqVisIdxOf[corrIdx];
            if (vIdx >= 0 && (size_t)(vIdx*3+2) < curVisVerts.size()) {
                return glm::vec3(curVisVerts[vIdx*3], curVisVerts[vIdx*3+1], curVisVerts[vIdx*3+2]);
            }
        }
        return st.correspondences[corrIdx].srcPos;
    };

    auto drawArrow = [&](const glm::vec3& a, const glm::vec3& b,
                         const glm::vec3& cA, const glm::vec3& cB,
                         int nSeg, float r) {
        for (int s = 1; s < nSeg; s++) {
            float t = static_cast<float>(s) / static_cast<float>(nSeg);
            glm::vec3 mid = glm::mix(a, b, t);
            glm::vec3 col = glm::mix(cA, cB, t);
            sphere.draw(shader, mid, col, r, view, proj, camPos);
        }
    };

    int activeSlot = ctrl.activeSlot();

    std::vector<glm::vec3> placedPositions;
    placedPositions.reserve(N);
    for (int i = 0; i < N; i++) {
        if (i == activeSlot) continue;
        auto state = ctrl.slotState(i);
        float prog = ctrl.slotProgress(i);
        if (state == SequentialDeformController::SLOT_UNTOUCHED) continue;
        if (state == SequentialDeformController::SLOT_EDITED && prog <= 1e-5f) continue;
        int corrIdx = st.seqMoveQueue[i];
        glm::vec3 center = getCenter(corrIdx);
        const auto& c = st.correspondences[corrIdx];
        if (state == SequentialDeformController::SLOT_SKIPPED) {
            sphere.draw(shader, center,   colSkipped, radius * 0.30f, view, proj, camPos);
            sphere.draw(shader, c.tgtPos, colSkipTgt, radius * 0.22f, view, proj, camPos);
        } else {
            sphere.draw(shader, center,   colEdited,  radius * 0.55f, view, proj, camPos);
            sphere.draw(shader, c.tgtPos, colEditTgt, radius * 0.38f, view, proj, camPos);
        }
        placedPositions.push_back(center);
    }
    if (activeSlot >= 0 && activeSlot < N) {
        placedPositions.push_back(getCenter(st.seqMoveQueue[activeSlot]));
    }

    float minSep2 = minSeparation * minSeparation;
    for (int i = loIdx; i <= hiIdx; i++) {
        if (i == activeSlot) continue;
        auto state = ctrl.slotState(i);
        float prog = ctrl.slotProgress(i);
        bool isEffectivelyUntouched =
            (state == SequentialDeformController::SLOT_UNTOUCHED) ||
            (state == SequentialDeformController::SLOT_EDITED && prog <= 1e-5f);
        if (!isEffectivelyUntouched) continue;
        int corrIdx = st.seqMoveQueue[i];
        glm::vec3 center = getCenter(corrIdx);
        if (minSeparation > 0.0f) {
            bool tooClose = false;
            for (const auto& p : placedPositions) {
                glm::vec3 dv = center - p;
                if (glm::dot(dv, dv) < minSep2) { tooClose = true; break; }
            }
            if (tooClose) continue;
        }
        placedPositions.push_back(center);
        const auto& c = st.correspondences[corrIdx];
        sphere.draw(shader, center,   colPlanned, radius * 0.50f, view, proj, camPos);
        sphere.draw(shader, c.tgtPos, colPlanTgt, radius * 0.35f, view, proj, camPos);
        drawArrow(center, c.tgtPos, colPlanned, colPlanTgt, 4, radius * 0.15f);
    }

    if (activeSlot >= 0 && activeSlot < N && st.seqHasLastMove) {
        glm::vec3 curCenter = st.seqLastMove.center;
        if (st.seqLastCorrIdx >= 0 && st.seqLastCorrIdx < (int)st.seqVisIdxOf.size()) {
            int v = st.seqVisIdxOf[st.seqLastCorrIdx];
            if (v >= 0 && (size_t)(v*3+2) < curVisVerts.size()) {
                curCenter = glm::vec3(curVisVerts[v*3], curVisVerts[v*3+1], curVisVerts[v*3+2]);
            }
        }
        sphere.draw(shader, curCenter,            colCurrent, radius * 1.0f, view, proj, camPos);
        sphere.draw(shader, st.seqLastMove.target, colCurTgt, radius * 0.75f, view, proj, camPos);
        drawArrow(curCenter, st.seqLastMove.target, colCurrent, colCurTgt, 6, radius * 0.20f);
    }
}

#endif
