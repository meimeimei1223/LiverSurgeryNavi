#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "FullSphereCameraWithTarget.h"
#include "Sphere.h"
#include "AR.h"
#include "OBJTargetExtraction.h"
#include "RegistrationCore.h"
class mCutMesh;
struct ImageSessionState {
    std::string path;
    int         width  = 0;
    int         height = 0;
    bool        loaded = false;
};
struct MaskPoint {
    float u  = 0.0f;
    float v  = 0.0f;
    bool  fg = true;
};

// Which segmentation the user is currently editing.
//   Liver       : the primary organ mask (segmentation_mask.png).
//   Instrument  : a secondary mask of occluding tools / instruments
//                 (instrument_segmentation_mask.png), used downstream
//                 for boundary rejection during registration.
// Stored on AppContext::activeMaskKind so the click handler knows which
// vector to push into and the renderer knows which color to use.
enum class MaskKind {
    Liver = 0,
    Instrument = 1,
};
enum class AppMode {
    kEmpty,
    kImageOnly,      // 通常の画像表示モード
    kMaskSelection,  // マスク選択専用モード（固定アスペクト比）
    kRegistration
};
struct AppContext {
    GLFWwindow* window  = nullptr;
    int         windowW = 1280;
    int         windowH = 720;
    glm::mat4 model{1.0f};
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::vec3 objPos{0.0f};
    FullSphereCamera orbitCam;
    mCutMesh* liver   = nullptr;
    mCutMesh* portal  = nullptr;
    mCutMesh* vein    = nullptr;
    mCutMesh* tumor   = nullptr;
    mCutMesh* segment = nullptr;
    mCutMesh* gb      = nullptr;
    mCutMesh*  screen = nullptr;
    SphereMesh deformMarker;
    SphereMesh registrationMarker;
    std::vector<mCutMesh*> all;
    std::vector<float>     alphas{0.8f, 0.9f, 0.9f, 0.5f, 0.5f, 0.7f, 0.5f};
    RegistrationData              reg;
    Reg3DCustom::CameraIntrinsics intrinsics;
    std::string                   objSourcePath;
    glm::vec3                     lastOrganOffset{0.0f};
    ImageSessionState      image;
    AppMode                mode = AppMode::kEmpty;
    // Liver-mask points. Kept under the original name for backwards-compat
    // with existing callers (DepthRunner / RunDepth path / MaskPicker undo).
    std::vector<MaskPoint> maskPoints;
    // Instrument-mask points. Populated only while activeMaskKind ==
    // Instrument. Empty by default; downstream code treats "empty" as
    // "no instrument mask requested" and skips the second SAM2 pass.
    std::vector<MaskPoint> instrumentMaskPoints;
    // Which mask the next click goes into, and which one Segment-1 /
    // Instrument buttons preview. Default = Liver so the existing
    // single-mask UX is unchanged for users who don't touch Instrument.
    MaskKind               activeMaskKind = MaskKind::Liver;
    AR::Background arBg;
    bool  arMode                 = false;
    float silhouetteCosThreshold = 0.10f;
    bool      dragging = false;
    int       hitIndex = -1;
    glm::vec3 hitPos{0.0f};

    // Whether the next Run Depth / Instrument preview should auto-detect
    // the FOV vignette and OR-merge it into the occluder mask. ON by
    // default = standard production behaviour. The DEPTH GENERATION panel
    // exposes a checkbox so the user can toggle this BEFORE running, since
    // the decision is baked into instrument_segmentation_mask.png at that
    // moment and cannot be changed afterward without rerunning the pipeline.
    bool detectVignette = true;

    // Whether to run the depth pipeline (SAM2 + DA3) on GPU via CUDA.
    // Toggled by the "Use GPU" checkbox in the DEPTH GENERATION panel.
    // Default OFF for safety on machines without CUDA; flip ON to get
    // ~4-5x faster Run Depth when a CUDA GPU is available AND the
    // sam2_da3_lite binary was built with USE_CUDA=ON.
    bool useCuda = false;

};
