#pragma once
/*
 * RefineSyncTypes.h
 * -----------------
 * Small shared types needed by both main.cpp's synchronous Refine
 * runner and the Shift+P protocol runner (ProtocolRunner.h).
 *
 * Keep this header minimal — just the POD types. Actual logic
 * lives in main.cpp (runRefineSync) and ProtocolRunner.h.
 */

#ifndef REFINE_SYNC_TYPES_H
#define REFINE_SYNC_TYPES_H

/* Result of a synchronous Refine invocation (see runRefineSync
 * in main.cpp). All fields are populated by the runner; callers
 * inspect them read-only. */
struct RefineSyncResult {
    float initialRmse    = 0.0f;   /* compRMSE right after init */
    float finalRmse      = 0.0f;   /* compRMSE after function returns
                                    * (= initialRmse when no improvement) */
    bool  improved       = false;  /* true iff finalRmse < initialRmse
                                    * Mesh is kept at best state when true,
                                    * restored to initial state when false. */
    int   iterationsRun  = 0;      /* actual number of refineStep calls executed */
    int   bestIteration  = 0;      /* iteration index that produced bestRMSE */
    float elapsedSec     = 0.0f;   /* wall-clock for init + loop + finalize */
};

#endif /* REFINE_SYNC_TYPES_H */
