#ifndef AUTO_HANDLE_CONTROLLER_H
#define AUTO_HANDLE_CONTROLLER_H

#include "HandleControllerBase.h"
#include "AutoDeform.h"
#include "Sphere.h"
#include "ShaderProgram.h"

class AutoHandleController : public HandleControllerBase {
public:
    bool initialize(SoftBody* body, const AutoDeform::State& st) {
        clear(body);
        if (!body) return false;

        for (const auto& fh : st.fixHandles) {
            Handle h;
            if (registerHandle_(body, fh.center, glm::vec3(0.0f), fh.radius, /*isFixed=*/true, h)) {
                fixHandles_.push_back(std::move(h));
            }
        }

        for (const auto& mh : st.moveHandles) {
            glm::vec3 dir = mh.target - mh.center;
            Handle h;
            if (registerHandle_(body, mh.center, dir, mh.radius, /*isFixed=*/false, h)) {
                moveHandles_.push_back(std::move(h));
            }
        }
        activeMoveIdx_ = moveHandles_.empty() ? -1 : 0;
        std::cout << "[AutoHandleCtrl] initialized: fix=" << fixHandles_.size()
                  << " move=" << moveHandles_.size() << std::endl;
        return true;
    }
};

inline void drawAutoHandles(
    const AutoHandleController& ctrl,
    SphereMesh& sphere,
    ShaderProgram& shader,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& camPos,
    float fixRadiusUnit,
    float moveRadiusUnit)
{
    static const glm::vec3 colFix     (0.20f, 0.50f, 1.00f);
    static const glm::vec3 colUnt     (1.00f, 1.00f, 0.10f);
    static const glm::vec3 colEdit    (0.10f, 0.85f, 0.30f);
    static const glm::vec3 colActive  (1.00f, 0.55f, 0.00f);
    static const glm::vec3 colTgt     (1.00f, 0.10f, 0.10f);

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
