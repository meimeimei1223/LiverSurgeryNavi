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
        glm::vec3 newCenter = h.initialCenter + h.dirVec * h.progress;
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
        for (size_t i = 0; i < fixHandles_.size(); i++) {
            Handle& h = fixHandles_[i];
            for (size_t k = 0; k < h.vertices.size(); k++) {
                int vi = h.vertices[k];
                glm::vec3 pos = h.initialCenter + h.relativePositions[k];
                body->setParticlePosition(vi, pos);
                body->setParticlePrevPosition(vi, pos);
                body->setParticleVelocity(vi, glm::vec3(0.0f));
            }
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
