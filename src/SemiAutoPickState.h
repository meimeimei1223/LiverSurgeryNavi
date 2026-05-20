#ifndef SEMI_AUTO_PICK_STATE_H
#define SEMI_AUTO_PICK_STATE_H

#include <vector>
#include <glm/glm.hpp>

enum SemiPickPhase {
    SEMI_OFF       = 0,
    SEMI_PICK_FIX  = 1,
    SEMI_PICK_PAIRS= 2,
    SEMI_EDIT      = 3
};

enum SemiPickKind {
    SEMI_KIND_TGT = 0,
    SEMI_KIND_SRC = 1
};

struct SemiAutoPickData {
    SemiPickPhase phase = SEMI_OFF;

    int targetFixCount  = 4;
    int targetPairCount = 4;

    float fixRadius     = 0.5f;
    float moveRadius    = 0.5f;

    std::vector<glm::vec3>    fixPoints;
    std::vector<glm::vec3>    tgtList;
    std::vector<glm::vec3>    srcList;
    std::vector<SemiPickKind> pickOrder;

    void clearAll() {
        phase = SEMI_OFF;
        fixPoints.clear();
        tgtList.clear();
        srcList.clear();
        pickOrder.clear();
    }
};

#endif
