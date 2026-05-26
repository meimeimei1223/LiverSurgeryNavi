#ifndef SEMI_AUTO_HANDLE_CONTROLLER_H
#define SEMI_AUTO_HANDLE_CONTROLLER_H

#include "HandleControllerBase.h"
#include "Sphere.h"
#include "ShaderProgram.h"

class SemiAutoHandleController : public HandleControllerBase {
public:
    bool initialize(SoftBody* body) {
        clear(body);
        if (!body) return false;
        activeMoveIdx_ = -1;
        std::cout << "[SemiAutoHandleCtrl] initialized (empty)" << std::endl;
        return true;
    }

    int addMovePair(SoftBody* body,
                    const glm::vec3& srcWorld,
                    const glm::vec3& tgtWorld,
                    float radius)
    {
        if (!body) return -1;
        glm::vec3 dir = tgtWorld - srcWorld;
        Handle h;
        if (!registerHandle_(body, srcWorld, dir, radius, false, h)) return -1;
        moveHandles_.push_back(std::move(h));
        int idx = static_cast<int>(moveHandles_.size()) - 1;
        if (activeMoveIdx_ < 0) activeMoveIdx_ = idx;
        std::cout << "[SemiAutoHandleCtrl] addMovePair idx=" << idx
                  << " |disp|=" << glm::length(dir)
                  << " radius=" << radius
                  << " verts=" << moveHandles_.back().vertices.size() << std::endl;
        return idx;
    }

    int addFixHandle(SoftBody* body,
                     const glm::vec3& srcWorld,
                     float radius)
    {
        if (!body) return -1;
        Handle h;
        if (!registerHandle_(body, srcWorld, glm::vec3(0.0f), radius, true, h)) return -1;
        fixHandles_.push_back(std::move(h));
        int idx = static_cast<int>(fixHandles_.size()) - 1;
        std::cout << "[SemiAutoHandleCtrl] addFixHandle idx=" << idx
                  << " radius=" << radius
                  << " verts=" << fixHandles_.back().vertices.size() << std::endl;
        return idx;
    }

    bool removeMove(SoftBody* body, int idx) {
        if (!body) return false;
        if (idx < 0 || idx >= static_cast<int>(moveHandles_.size())) return false;
        restoreUnsharedVertices_(body, moveHandles_[idx]);
        moveHandles_.erase(moveHandles_.begin() + idx);
        if (moveHandles_.empty()) {
            activeMoveIdx_ = -1;
        } else {
            if (activeMoveIdx_ >= static_cast<int>(moveHandles_.size()))
                activeMoveIdx_ = static_cast<int>(moveHandles_.size()) - 1;
            if (activeMoveIdx_ < 0) activeMoveIdx_ = 0;
        }
        std::cout << "[SemiAutoHandleCtrl] removeMove idx=" << idx
                  << " remaining=" << moveHandles_.size() << std::endl;
        return true;
    }

    bool removeFix(SoftBody* body, int idx) {
        if (!body) return false;
        if (idx < 0 || idx >= static_cast<int>(fixHandles_.size())) return false;
        restoreUnsharedVertices_(body, fixHandles_[idx]);
        fixHandles_.erase(fixHandles_.begin() + idx);
        std::cout << "[SemiAutoHandleCtrl] removeFix idx=" << idx
                  << " remaining=" << fixHandles_.size() << std::endl;
        return true;
    }

private:
    void restoreUnsharedVertices_(SoftBody* body, const Handle& victim) {
        if (!body) return;
        for (int vi : victim.vertices) {
            if (isVertexOwnedByOther_(vi, &victim)) continue;
            for (size_t s = 0; s < savedIdx_.size(); s++) {
                if (savedIdx_[s] == vi) {
                    body->setInvMass(vi, savedInvMasses_[s]);
                    savedIdx_.erase(savedIdx_.begin() + s);
                    savedInvMasses_.erase(savedInvMasses_.begin() + s);
                    break;
                }
            }
        }
    }

    bool isVertexOwnedByOther_(int vi, const Handle* exclude) const {
        for (const auto& h : fixHandles_) {
            if (&h == exclude) continue;
            for (int v : h.vertices) if (v == vi) return true;
        }
        for (const auto& h : moveHandles_) {
            if (&h == exclude) continue;
            for (int v : h.vertices) if (v == vi) return true;
        }
        return false;
    }
};

inline void drawSemiAutoHandles(
    const SemiAutoHandleController& ctrl,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    float fixRadiusUnit,
    float moveRadiusUnit)
{
    static const glm::vec3 colFix    (0.20f, 0.50f, 1.00f);
    static const glm::vec3 colUnt    (1.00f, 1.00f, 0.10f);
    static const glm::vec3 colEdit   (0.10f, 0.85f, 0.30f);
    static const glm::vec3 colActive (1.00f, 0.55f, 0.00f);
    static const glm::vec3 colTgt    (1.00f, 0.10f, 0.10f);

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

    for (int i = 0; i < ctrl.numFix(); i++) {
        const auto& h = ctrl.fixHandle(i);
        sphere.draw(shader, h.initialCenter, colFix, fixRadiusUnit, view, proj, camPos);
    }

    for (int i = 0; i < ctrl.numMove(); i++) {
        const auto& h = ctrl.moveHandle(i);
        glm::vec3 curCenter = h.initialCenter + h.dirVec * h.progress;
        glm::vec3 tgtFinal  = h.initialCenter + h.dirVec;
        bool isActive = (i == ctrl.activeMoveIdx());
        glm::vec3 centerCol;
        float rScale;
        if (isActive) {
            centerCol = colActive;
            rScale = 1.0f;
        } else if (h.progress > 1e-5f) {
            centerCol = colEdit;
            rScale = 0.7f;
        } else {
            centerCol = colUnt;
            rScale = 0.7f;
        }
        sphere.draw(shader, curCenter, centerCol, moveRadiusUnit * rScale, view, proj, camPos);
        sphere.draw(shader, tgtFinal,  colTgt,    moveRadiusUnit * rScale * 0.55f, view, proj, camPos);
        drawArrow(curCenter, tgtFinal, centerCol, colTgt, 6, moveRadiusUnit * rScale * 0.18f);
    }
}

#endif
