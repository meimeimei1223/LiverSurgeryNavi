#pragma once

// =============================================================================
//  StlExport.h — registered-mesh export, factored out of the old GLFW_KEY_M
//  switch case (key-reorg Phase 12).
//
//  The two functions below were the plain-M and Shift+M branches. Their bodies
//  touch a large set of main.cpp globals (the six organ meshes, REG_MODEL_PATH /
//  DEPTH_OUTPUT_PATH / Reg_*_FILE_PATH, g_originalLiverDiagMm / g_intrinsicsSource,
//  timestamp + filesystem helpers, ...), so the DEFINITIONS live in main.cpp
//  where all of that is in scope — declaring them here keeps a clean interface
//  for the UI Export buttons (onExportStl / onExportStlFlipped) without a large,
//  fragile extern block.
// =============================================================================

namespace StlExport {

// Plain M: export the registered organ meshes as OBJ to REG_MODEL_PATH
// (reg_liver.obj, reg_portal.obj, ... — the REG -> DEFORM handoff).
void exportRegisteredObjs();

// Shift+M: export cam-mm STL (X+Z flip, CT-truth scale) for tumor/liver and
// write a reproducibility snapshot of the Run-Depth inputs + STL outputs.
void exportCamMmStlWithSnapshot();

} // namespace StlExport
