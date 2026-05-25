// =============================================================================
//  OBJTargetExtraction.h
//  ---------------------------------------------------------------------------
//  OBJ-based target point cloud extraction for registration pipeline.
//
//  Replaces the grid-mesh depth-texture pipeline with a direct OBJ loader
//  that reads a metric point-cloud OBJ (e.g. from DepthAnything + pinhole
//  unprojection) and produces a PointCloud ready for registration.
//
//  Integration model:
//      1. main.cpp loads the OBJ into screenMesh (via mCutMesh::loadMeshFromFile)
//      2. main.cpp calls  extractTargetFromOBJ(...)   -> PointCloud
//      3. main.cpp calls  setCachedTargetCloud(cloud)
//      4. Existing 15+ call sites to extractFrontFacePoints()
//         transparently receive the cached cloud instead of running the
//         grid code path.  No call site changes needed.
//
//  Coordinate conventions:
//      Target OBJ is in camera-space, metric (meters), with
//        OpenGL convention  (+X right, +Y UP, +Z forward).
//      For mask lookup we reproject to image pixels, flipping Y sign.
//      The 3D coordinates stored in the PointCloud keep the original
//      OpenGL convention -- no axis change, only the projection step
//      flips Y.
//
//  Verified against data:   depth_output/pc_metric_pinhole_masked_k4a.obj
//  Diagnostic tool:         diagnose_obj.cpp (100% mask-alignment PASS)
// =============================================================================

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include <cmath>

#include <glm/glm.hpp>

#include "mCutMesh.h"
#include "NoOpen3DRegistration.h"   // PointCloud + cache accessors
#include "DepthUtils.h"             // g_boundaryDistMap, ensureBoundaryMap

namespace Reg3DCustom {

// =============================================================================
//  Camera intrinsics (pinhole)
// -----------------------------------------------------------------------------
//  Convention
//      OpenCV / Kinect / DepthAnything: cx, cy measured from image TOP-LEFT.
//      Focal length in pixels. No distortion coefficients stored here --
//      the app assumes the input RGB / depth / OBJ were produced from a
//      RECTIFIED image. Any distortion correction happens upstream,
//      in whatever pipeline unprojects the depth map into the OBJ.
//
//  Extending for a new camera
//      1. Run a calibration of your choice at the target resolution
//         (Zhang's method with a checkerboard, AprilTag, ChArUco, etc.
//         -- any tool that yields pinhole fx/fy/cx/cy will do).
//      2. If the calibration also yields distortion coefficients, apply
//         them in your depth-unprojection code before writing the OBJ;
//         do NOT pass them here.
//      3. Write the 6 values (fx, fy, cx, cy, width, height) into a text
//         file -- format documented at loadCameraIntrinsics below -- OR
//         register a static factory method on this struct. Factories are
//         the zero-file fallback; files are the runtime-configurable path.
//      4. For SDKs that expose intrinsics at runtime (k4a, librealsense),
//         you can read them once and dump via saveCameraIntrinsics below.
// =============================================================================
struct CameraIntrinsics {
    float       fx = 0.0f, fy = 0.0f;
    float       cx = 0.0f, cy = 0.0f;
    int         width = 0, height = 0;
    std::string name;                       // optional identifier, free-form

    // Distortion (Brown-Conrady, OpenCV convention). All zero -> no distortion,
    // image is treated as already-rectified. See Undistort.h for the math
    // and undistortion pipeline. k4 is the 8th-order radial term (rare; many
    // calibration tools omit it -- safe to leave 0).
    float k1 = 0.0f, k2 = 0.0f, k3 = 0.0f, k4 = 0.0f;
    float p1 = 0.0f, p2 = 0.0f;

    bool valid() const {
        return fx > 0.0f && fy > 0.0f && width > 0 && height > 0;
    }

    // True iff any distortion coefficient is non-trivial. Threshold 1e-6 lets
    // explicit "0.0" values in calibration files round-trip cleanly.
    bool hasDistortion() const {
        const float eps = 1e-6f;
        return std::fabs(k1) > eps || std::fabs(k2) > eps ||
               std::fabs(k3) > eps || std::fabs(k4) > eps ||
               std::fabs(p1) > eps || std::fabs(p2) > eps;
    }

    // --- Factory: Azure Kinect Color ----------------------------------------
    static CameraIntrinsics k4a_color_720p()  {
        return { 918.234f, 918.112f, 640.152f, 366.447f,
                1280,     720,      "azure_kinect_color_720p" };
    }
    static CameraIntrinsics k4a_color_1080p() {
        // 1080p = 720p * 1.5; principal point scales linearly
        return { 1377.35f, 1377.17f, 960.228f, 549.671f,
                1920,     1080,     "azure_kinect_color_1080p" };
    }

    // --- Factory: Intel RealSense -------------------------------------------
    // Typical factory-calibrated values; your unit will differ slightly.
    static CameraIntrinsics d435_color_720p() {
        return { 919.787f, 919.418f, 645.250f, 362.810f,
                1280,     720,      "realsense_d435_color_720p" };
    }
    static CameraIntrinsics d435_color_480p() {
        return { 613.192f, 612.945f, 323.500f, 241.874f,
                848,      480,      "realsense_d435_color_480p" };
    }
    static CameraIntrinsics d455_color_720p() {
        return { 635.001f, 635.187f, 636.527f, 353.945f,
                1280,     720,      "realsense_d455_color_720p" };
    }

    // --- Factory: generic pinhole from resolution ---------------------------
    // Uses a 55 deg vertical FOV (approx smartphone rear camera).  Useful
    // as a sanity default when no calibration file is available.
    static CameraIntrinsics pinhole_default(int W, int H) {
        const float vfov_rad = 55.0f * 3.14159265358979f / 180.0f;
        const float fy = static_cast<float>(H) * 0.5f /
                         std::tan(vfov_rad * 0.5f);
        return { fy, fy, W * 0.5f, H * 0.5f, W, H, "generic_pinhole_55vfov" };
    }
};

// ============================================================
//  Y-axis sign convention for the OBJ source
// ------------------------------------------------------------
//  OPENCV  (+1):  +Y points DOWN  (raw depth-map unprojection)
//  OPENGL  (-1):  +Y points UP    (DepthAnything / sam2_da3)
//  Applied only when projecting to image pixels for mask lookup.
//  3D coordinates in the returned PointCloud are NEVER flipped.
// ============================================================
inline constexpr float OBJ_Y_SIGN_OPENGL = -1.0f;
inline constexpr float OBJ_Y_SIGN_OPENCV = +1.0f;

// =============================================================================
//  Load intrinsics from a text file. Format (one "key value" per line):
//
//      # any line starting with '#' is a comment
//      name    my_camera     # optional, free-form identifier
//      fx      918.234
//      fy      918.112
//      cx      640.152       # from image TOP-LEFT (OpenCV convention)
//      cy      366.447       # from image TOP-LEFT (OpenCV convention)
//      width   1280
//      height  720
//
//      # Optional Brown-Conrady distortion coefficients. All optional --
//      # absent values default to 0 (= no distortion in that term).
//      k1      -0.20         # radial r^2
//      k2       0.26         # radial r^4
//      k3      -0.12         # radial r^6
//      k4       0.0          # radial r^8 (rare)
//      p1       0.0          # tangential
//      p2       0.0          # tangential
//
//  Keys are case-insensitive. Order is free. Unknown keys are ignored so
//  the file can carry extra metadata (date, operator, etc.).
// =============================================================================
inline bool loadCameraIntrinsics(const std::string& path, CameraIntrinsics& K) {
    std::ifstream f(path);
    if (!f.is_open()) {
        // Don't log as error -- this is routinely called speculatively
        // inside loadCameraIntrinsicsAny() to probe candidate filenames.
        return false;
    }

    auto toLower = [](std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(c));
        return s;
    };

    std::string line;
    while (std::getline(f, line)) {
        // Strip trailing CR for Windows files
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Skip blanks and comments
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        if (line[first] == '#')         continue;

        std::istringstream iss(line);
        std::string key;
        if (!(iss >> key)) continue;
        key = toLower(key);

        if      (key == "fx")     iss >> K.fx;
        else if (key == "fy")     iss >> K.fy;
        else if (key == "cx")     iss >> K.cx;
        else if (key == "cy")     iss >> K.cy;
        else if (key == "width")  iss >> K.width;
        else if (key == "height") iss >> K.height;
        else if (key == "name")   iss >> K.name;
        else if (key == "k1")     iss >> K.k1;
        else if (key == "k2")     iss >> K.k2;
        else if (key == "k3")     iss >> K.k3;
        else if (key == "k4")     iss >> K.k4;
        else if (key == "p1")     iss >> K.p1;
        else if (key == "p2")     iss >> K.p2;
        // Other keys (date, operator, camera_model) are silently ignored
    }

    if (!K.valid()) {
        std::cerr << "[CameraIntrinsics] incomplete or invalid: "
                  << path << std::endl;
        return false;
    }
    std::cout << "[CameraIntrinsics] loaded "
              << (K.name.empty() ? "(unnamed)" : K.name)
              << "  fx=" << K.fx << " fy=" << K.fy
              << " cx=" << K.cx << " cy=" << K.cy
              << "  " << K.width << "x" << K.height
              << "  (" << path << ")" << std::endl;
    if (K.hasDistortion()) {
        std::cout << "[CameraIntrinsics] distortion"
                  << "  k1=" << K.k1 << " k2=" << K.k2
                  << " k3=" << K.k3 << " k4=" << K.k4
                  << " p1=" << K.p1 << " p2=" << K.p2 << std::endl;
    }
    return true;
}

// =============================================================================
//  Try each candidate path in order; return the first one that loads.
//  Useful when the camera source is ambiguous at build time: e.g. try a
//  user-named file, then a camera-specific file, then a generic one.
// =============================================================================
inline bool loadCameraIntrinsicsAny(
    const std::vector<std::string>& candidates,
    CameraIntrinsics& K)
{
    for (const auto& path : candidates) {
        if (loadCameraIntrinsics(path, K)) return true;
    }
    std::cerr << "[CameraIntrinsics] none of "
              << candidates.size() << " candidate files loaded" << std::endl;
    return false;
}

// =============================================================================
//  Verify that intrinsics resolution matches the expected image size.
//  The mask reprojection inside extractTargetFromOBJ uses K.width/height
//  as the pixel grid, so any mismatch with the real RGB/mask image leads
//  to silent projection drift. Log loudly if it happens.
// =============================================================================
inline bool checkIntrinsicsResolution(const CameraIntrinsics& K,
                                      int expectedW, int expectedH)
{
    if (K.width == expectedW && K.height == expectedH) return true;
    std::cerr << "[CameraIntrinsics] RESOLUTION MISMATCH: intrinsics are for "
              << K.width << "x" << K.height
              << " but the image is " << expectedW << "x" << expectedH
              << ". Projection will be wrong." << std::endl;
    return false;
}

// =============================================================================
//  Save intrinsics in the standard text format. Useful for persisting
//  values obtained from a camera SDK or the user's own calibration code.
//
//  Example -- dump once, then every future run reads the file:
//      Reg3DCustom::CameraIntrinsics K;
//      K.fx = myCalib.fx; K.fy = myCalib.fy;
//      K.cx = myCalib.cx; K.cy = myCalib.cy;
//      K.width  = imageW; K.height = imageH;
//      K.name   = "my_camera_self_calibrated";
//      Reg3DCustom::saveCameraIntrinsics(
//          DEPTH_OUTPUT_PATH + "intrinsics.txt", K);
//
//  (Source can be a custom Zhang implementation, a vendor SDK, or any
//   other pinhole calibrator -- this function just writes the numbers.)
// =============================================================================
inline bool saveCameraIntrinsics(const std::string& path,
                                 const CameraIntrinsics& K)
{
    if (!K.valid()) {
        std::cerr << "[CameraIntrinsics] refuse to save invalid intrinsics: "
                  << path << std::endl;
        return false;
    }
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[CameraIntrinsics] cannot write: " << path << std::endl;
        return false;
    }
    f << "# Pinhole intrinsics -- OpenCV convention (cx/cy from top-left)\n"
      << "# Saved by Reg3DCustom::saveCameraIntrinsics\n";
    if (!K.name.empty()) f << "name   " << K.name << "\n";
    f << "fx     " << K.fx     << "\n"
      << "fy     " << K.fy     << "\n"
      << "cx     " << K.cx     << "\n"
      << "cy     " << K.cy     << "\n"
      << "width  " << K.width  << "\n"
      << "height " << K.height << "\n";
    if (K.hasDistortion()) {
        f << "# Brown-Conrady distortion (radial + tangential)\n"
          << "k1     " << K.k1 << "\n"
          << "k2     " << K.k2 << "\n"
          << "k3     " << K.k3 << "\n"
          << "k4     " << K.k4 << "\n"
          << "p1     " << K.p1 << "\n"
          << "p2     " << K.p2 << "\n";
    }
    std::cout << "[CameraIntrinsics] wrote " << path << std::endl;
    return true;
}

// =============================================================================
//  Compute per-vertex normals from face indices
//      mCutMesh::loadMeshFromFile does not populate mNormals.
//      We average face normals of all triangles using each vertex.
//      Normals follow the same coordinate frame as the vertices (Y-up).
// =============================================================================
inline void computeVertexNormalsFromFaces(mCutMesh& mesh) {
    const size_t nV = mesh.mVertices.size() / 3;
    const size_t nI = mesh.mIndices.size();
    if (nV == 0 || nI < 3) {
        std::cerr << "[Normals] mesh has no faces -- skipping normal computation"
                  << std::endl;
        return;
    }

    mesh.mNormals.assign(nV * 3, 0.0f);

    auto vget = [&](GLuint idx) -> glm::vec3 {
        return glm::vec3(mesh.mVertices[idx * 3 + 0],
                         mesh.mVertices[idx * 3 + 1],
                         mesh.mVertices[idx * 3 + 2]);
    };

    for (size_t i = 0; i + 2 < nI; i += 3) {
        GLuint a = mesh.mIndices[i];
        GLuint b = mesh.mIndices[i + 1];
        GLuint c = mesh.mIndices[i + 2];
        glm::vec3 v0 = vget(a), v1 = vget(b), v2 = vget(c);
        glm::vec3 fn = glm::cross(v1 - v0, v2 - v0);
        // accumulate non-normalized face normal (area-weighted)
        mesh.mNormals[a*3+0] += fn.x; mesh.mNormals[a*3+1] += fn.y; mesh.mNormals[a*3+2] += fn.z;
        mesh.mNormals[b*3+0] += fn.x; mesh.mNormals[b*3+1] += fn.y; mesh.mNormals[b*3+2] += fn.z;
        mesh.mNormals[c*3+0] += fn.x; mesh.mNormals[c*3+1] += fn.y; mesh.mNormals[c*3+2] += fn.z;
    }

    // normalize each
    int zeroCount = 0;
    for (size_t v = 0; v < nV; v++) {
        float nx = mesh.mNormals[v*3+0];
        float ny = mesh.mNormals[v*3+1];
        float nz = mesh.mNormals[v*3+2];
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len < 1e-12f) {
            mesh.mNormals[v*3+0] = 0.0f;
            mesh.mNormals[v*3+1] = 0.0f;
            mesh.mNormals[v*3+2] = 1.0f;
            zeroCount++;
        } else {
            float inv = 1.0f / len;
            mesh.mNormals[v*3+0] = nx * inv;
            mesh.mNormals[v*3+1] = ny * inv;
            mesh.mNormals[v*3+2] = nz * inv;
        }
    }
    std::cout << "[Normals] computed for " << nV << " vertices"
              << (zeroCount > 0 ? " (" + std::to_string(zeroCount) + " degenerate)" : "")
              << std::endl;
}

// =============================================================================
//  Extraction statistics (for logging / diagnostics)
// =============================================================================
struct OBJExtractionStats {
    int kept         = 0;
    int z_filtered   = 0;   // outside [zNear, zFar]
    int out_of_frame = 0;   // reprojection fell outside image bounds
    int out_of_mask  = 0;   // reprojection fell outside mask
};

// =============================================================================
//  extractTargetFromOBJ -- main extraction entry point
// -----------------------------------------------------------------------------
//  Given an OBJ mesh in camera space (metric, Y-up), produce a PointCloud
//  suitable for ICP registration:
//     - points        : 3D coordinates (unchanged from OBJ, Y-up preserved)
//     - normals       : copied from mesh.mNormals if present, else empty
//     - boundaryDist  : distance to mask boundary (from g_boundaryDistMap)
//     - colors        : uniform green (same as the grid-based extractor)
//
//  Behaviour:
//     1. For each vertex, range-filter by z (zNear, zFar)
//     2. Reproject to pixel (u, v) using K and Y-flip convention
//     3. Look up boundary distance in g_boundaryDistMap
//     4. If reprojected outside mask, skip (shouldn't happen for masked OBJ)
//
//  Requirements:
//     ensureBoundaryMap() MUST have been called before -- otherwise nothing
//     gets through.  Callers should invoke it at set-up time.
// =============================================================================
inline std::shared_ptr<PointCloud> extractTargetFromOBJ(
    const mCutMesh& mesh,
    const CameraIntrinsics& K,
    float y_sign = OBJ_Y_SIGN_OPENGL,
    float zNear  = 0.1f,
    float zFar   = 5.0f,
    OBJExtractionStats* outStats = nullptr)
{
    auto cloud = std::make_shared<PointCloud>();

    if (!K.valid()) {
        std::cerr << "[extractTargetFromOBJ] invalid intrinsics -- aborting"
                  << std::endl;
        return cloud;
    }
    if (!g_boundaryDistMap.valid) {
        std::cerr << "[extractTargetFromOBJ] boundary map not built --"
                  << " call ensureBoundaryMap() first" << std::endl;
        return cloud;
    }

    const size_t nV = mesh.mVertices.size() / 3;
    const bool hasNormals = (mesh.mNormals.size() == mesh.mVertices.size());

    cloud->points.reserve(nV);
    cloud->boundaryDist.reserve(nV);
    if (hasNormals) cloud->normals.reserve(nV);

    // Instrument mask is optional. Use only when valid AND its dimensions
    // match the liver boundary map. Otherwise log a warning and disable
    // for this scene; the rest of the pipeline behaves as before.
    const bool useInstMap = g_instrumentDistMap.valid &&
        g_instrumentDistMap.width  == g_boundaryDistMap.width &&
        g_instrumentDistMap.height == g_boundaryDistMap.height;
    if (g_instrumentDistMap.valid && !useInstMap) {
        std::cout << "[InstrumentMask] size mismatch ("
                  << g_instrumentDistMap.width  << "x" << g_instrumentDistMap.height
                  << " vs boundary "
                  << g_boundaryDistMap.width << "x" << g_boundaryDistMap.height
                  << ") -- disabled for this scene" << std::endl;
    }
    if (useInstMap) cloud->instrumentDist.reserve(nV);

    OBJExtractionStats S;

    for (size_t i = 0; i < nV; i++) {
        float x = mesh.mVertices[i * 3 + 0];
        float y = mesh.mVertices[i * 3 + 1];
        float z = mesh.mVertices[i * 3 + 2];

        // z range filter (shouldn't fire for masked OBJ, but defensive)
        if (z <= 1e-6f || z < zNear || z > zFar) {
            S.z_filtered++;
            continue;
        }

        // reproject with Y-flip (OpenGL -> image pixel)
        float u  = K.fx *          x  / z + K.cx;
        float vv = K.fy * (y_sign * y) / z + K.cy;
        int pu = (int)std::lround(u);
        int pv = (int)std::lround(vv);

        if (pu < 0 || pu >= K.width || pv < 0 || pv >= K.height) {
            S.out_of_frame++;
            continue;
        }

        // look up boundary distance in the pre-computed map
        float bd = g_boundaryDistMap.data[(size_t)pv * g_boundaryDistMap.width + pu];
        if (bd >= 9000.0f) {
            S.out_of_mask++;
            continue;
        }

        cloud->points.push_back(glm::vec3(x, y, z));   // 3D coords preserved
        cloud->boundaryDist.push_back(bd);
        if (useInstMap) {
            float idd = g_instrumentDistMap.data[
                (size_t)pv * g_instrumentDistMap.width + pu];
            cloud->instrumentDist.push_back(idd);
        }
        if (hasNormals) {
            cloud->normals.push_back(glm::vec3(
                mesh.mNormals[i * 3 + 0],
                mesh.mNormals[i * 3 + 1],
                mesh.mNormals[i * 3 + 2]));
        }
        S.kept++;
    }

    // uniform green colors -- matches extractFrontFacePoints() behaviour
    cloud->colors.resize(cloud->points.size(), glm::vec3(0.0f, 1.0f, 0.0f));

    std::cout << "[extractTargetFromOBJ] kept=" << S.kept
              << "  z_filtered=" << S.z_filtered
              << "  out_of_frame=" << S.out_of_frame
              << "  out_of_mask=" << S.out_of_mask
              << "  (total=" << nV
              << ", normals=" << (hasNormals ? "YES" : "NO")
              << ", instDist=" << (useInstMap ? "YES" : "NO")
              << ")" << std::endl;

    if (outStats) *outStats = S;
    return cloud;
}

// =============================================================================
//  Convenience: full setup in one call
//      1. ensureBoundaryMap
//      2. computeVertexNormalsFromFaces (if missing)
//      3. extractTargetFromOBJ
//      4. setCachedTargetCloud
//  Returns the extracted cloud (also injected into the registration cache).
// =============================================================================
inline std::shared_ptr<PointCloud> setupOBJTarget(
    mCutMesh& objMesh,
    const CameraIntrinsics& K,
    float y_sign = OBJ_Y_SIGN_OPENGL,
    float zNear  = 0.1f,
    float zFar   = 5.0f)
{
    if (!ensureBoundaryMap()) {
        std::cerr << "[setupOBJTarget] ensureBoundaryMap failed --"
                  << " check DEPTH_OUTPUT_PATH/segmentation_mask.png" << std::endl;
        return nullptr;
    }
    // Optional: load instrument mask if present. Failure is non-fatal --
    // the boundary classifier simply skips the instrument-exclusion step.
    ensureInstrumentDistMap();
    if (objMesh.mNormals.size() != objMesh.mVertices.size()) {
        computeVertexNormalsFromFaces(objMesh);
    }
    auto cloud = extractTargetFromOBJ(objMesh, K, y_sign, zNear, zFar);
    setCachedTargetCloud(cloud);
    return cloud;
}

// =============================================================================
//  Uniform scale applied to both the display mesh and the cached point cloud.
//  The OBJ is metric (meters) but organ meshes typically use model units of
//  order 1-10; the old pipeline multiplied screenMesh by 10 to match. This
//  preserves that behaviour so organs remain visible at consistent scale.
//
//  Notes:
//      - boundaryDist is in PIXEL units and is NOT scaled
//      - normals keep direction under uniform positive scale (no change)
//      - call BEFORE Registration runs (ICP sees scaled target)
//      - call setUp(mesh) after to refresh GL buffers
// =============================================================================
inline void scaleMeshAndCloud(
    mCutMesh& mesh,
    PointCloud& cloud,
    float scale)
{
    if (scale == 1.0f) return;

    for (size_t i = 0; i < mesh.mVertices.size(); i++) {
        mesh.mVertices[i] *= scale;
    }
    for (auto& p : cloud.points) {
        p *= scale;
    }
    std::cout << "[scaleMeshAndCloud] applied scale=" << scale
              << "  (mesh verts=" << (mesh.mVertices.size() / 3)
              << ", cloud points=" << cloud.points.size() << ")" << std::endl;
}

// =============================================================================
//  Print axis-aligned bounding box of a mesh (for debugging)
// =============================================================================
inline void printMeshBBox(const mCutMesh& mesh, const std::string& label) {
    if (mesh.mVertices.empty()) {
        std::cout << "[BBox " << label << "] (empty mesh)" << std::endl;
        return;
    }
    float mn[3] = {  1e30f,  1e30f,  1e30f };
    float mx[3] = { -1e30f, -1e30f, -1e30f };
    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        for (int k = 0; k < 3; k++) {
            float v = mesh.mVertices[i + k];
            if (v < mn[k]) mn[k] = v;
            if (v > mx[k]) mx[k] = v;
        }
    }
    std::cout << "[BBox " << label << "]"
              << " X[" << mn[0] << ", " << mx[0] << "]"
              << " Y[" << mn[1] << ", " << mx[1] << "]"
              << " Z[" << mn[2] << ", " << mx[2] << "]"
              << std::endl;
}

// =============================================================================
//  Compute the AABB center of a mesh
// =============================================================================
inline glm::vec3 computeMeshCenter(const mCutMesh& mesh) {
    if (mesh.mVertices.empty()) return glm::vec3(0.0f);
    glm::vec3 mn( 1e30f,  1e30f,  1e30f);
    glm::vec3 mx(-1e30f, -1e30f, -1e30f);
    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        glm::vec3 v(mesh.mVertices[i], mesh.mVertices[i+1], mesh.mVertices[i+2]);
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    return (mn + mx) * 0.5f;
}

// =============================================================================
//  Translate both the display mesh and the cached point cloud by the same
//  offset. Use this to place the OBJ near the organ meshes so the camera
//  sees them together and ICP starts from a reasonable initial pose.
// =============================================================================
inline void translateMeshAndCloud(
    mCutMesh& mesh,
    PointCloud& cloud,
    const glm::vec3& offset)
{
    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        mesh.mVertices[i    ] += offset.x;
        mesh.mVertices[i + 1] += offset.y;
        mesh.mVertices[i + 2] += offset.z;
    }
    for (auto& p : cloud.points) {
        p += offset;
    }
    std::cout << "[translateMeshAndCloud] applied offset=("
              << offset.x << ", " << offset.y << ", " << offset.z
              << ")" << std::endl;
}

// =============================================================================
//  Convenience: center an OBJ (and its cloud) at the origin.
//  The OBJ from DepthAnything lives at Z~1m in camera space; after scaling
//  it would sit 10+ units in front of origin where the organ meshes live.
//  This moves it to where they can be seen together.
//
//  NOTE: Breaks the camera-space interpretation (no longer "where the real
//  liver physically is"). Prefer moveMeshesToTarget() for AR-friendly setup
//  that keeps the OBJ at its true camera-coord location.
// =============================================================================
inline void centerMeshAndCloudAtOrigin(mCutMesh& mesh, PointCloud& cloud) {
    glm::vec3 c = computeMeshCenter(mesh);
    translateMeshAndCloud(mesh, cloud, -c);
}

// =============================================================================
//  Move a set of meshes so their combined center aligns with a target mesh.
//  Used to bring organ models (liver, segment, ...) to the OBJ target's
//  position as a starting pose for ICP -- while the OBJ itself stays at its
//  physical camera-space location (Option B, AR-friendly).
//
//  Each mesh is rigidly translated; shape and local orientation preserved.
// =============================================================================
inline void moveMeshesToTarget(
    const std::vector<mCutMesh*>& meshes,
    const mCutMesh& target)
{
    if (meshes.empty() || target.mVertices.empty()) return;

    glm::vec3 targetCenter = computeMeshCenter(target);

    // Combined center (average of valid mesh centers)
    glm::vec3 meshesCenter(0.0f);
    int validCount = 0;
    for (auto* m : meshes) {
        if (m && !m->mVertices.empty()) {
            meshesCenter += computeMeshCenter(*m);
            validCount++;
        }
    }
    if (validCount == 0) return;
    meshesCenter /= (float)validCount;

    glm::vec3 offset = targetCenter - meshesCenter;

    for (auto* m : meshes) {
        if (!m) continue;
        for (size_t i = 0; i < m->mVertices.size(); i += 3) {
            m->mVertices[i    ] += offset.x;
            m->mVertices[i + 1] += offset.y;
            m->mVertices[i + 2] += offset.z;
        }
    }

    std::cout << "[moveMeshesToTarget] moved " << validCount << " mesh(es)"
              << " by (" << offset.x << ", " << offset.y << ", " << offset.z << ")"
              << "  toward target center ("
              << targetCenter.x << ", " << targetCenter.y << ", " << targetCenter.z
              << ")" << std::endl;
}

// =============================================================================
//  Compute axis-aligned bounding box diagonal of a mesh.
// =============================================================================
inline float computeMeshDiag(const mCutMesh& mesh) {
    if (mesh.mVertices.empty()) return 0.0f;
    glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
    for (size_t i = 0; i + 2 < mesh.mVertices.size(); i += 3) {
        glm::vec3 v(mesh.mVertices[i], mesh.mVertices[i+1], mesh.mVertices[i+2]);
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    return glm::length(mx - mn);
}

// =============================================================================
//  Pre-align a set of source meshes (organ models) to a target mesh by a
//  similarity transform: uniform scale + translation (no rotation).
//
//  Replaces moveMeshesToTarget() when source/target may differ in size.
//  The first mesh in `sources` (e.g. liverMesh3D) is used as the size
//  reference; all sources receive the SAME transform so their internal
//  geometry stays consistent.
//
//  Returned matrix T satisfies: p_new = T * p_old  (homogeneous).
//  Caller can store it for later undo (e.g. on rebuild).
//
//  Notes:
//      - Non-uniform scale is NOT applied (would distort organs)
//      - Rotation is NOT applied (orientation preserved -- registration
//        will handle that)
//      - If target diag or source diag is too small, scale is forced to 1
// =============================================================================
inline glm::mat4 prealignSourceToTarget(
    const std::vector<mCutMesh*>& sources,
    const mCutMesh& target,
    float minScaleChange = 0.02f)
{
    if (sources.empty() || !sources[0] || target.mVertices.empty())
        return glm::mat4(1.0f);

    // Reference: the first non-null source mesh
    mCutMesh* ref = nullptr;
    for (auto* m : sources) {
        if (m && !m->mVertices.empty()) { ref = m; break; }
    }
    if (!ref) return glm::mat4(1.0f);

    float srcDiag = computeMeshDiag(*ref);
    float tgtDiag = computeMeshDiag(target);
    if (srcDiag < 1e-6f || tgtDiag < 1e-6f) {
        std::cout << "[prealignSourceToTarget] degenerate diagonal, skipping"
                  << std::endl;
        return glm::mat4(1.0f);
    }

    float s = tgtDiag / srcDiag;
    if (std::abs(s - 1.0f) < minScaleChange) {
        // Diff small enough -> just translate (same as moveMeshesToTarget)
        s = 1.0f;
    }

    // Combined center across all valid sources (consistent with moveMeshesToTarget)
    glm::vec3 srcCenter(0.0f);
    int validCount = 0;
    for (auto* m : sources) {
        if (m && !m->mVertices.empty()) {
            srcCenter += computeMeshCenter(*m);
            validCount++;
        }
    }
    if (validCount == 0) return glm::mat4(1.0f);
    srcCenter /= (float)validCount;

    glm::vec3 tgtCenter = computeMeshCenter(target);

    // T = translate(tgtCenter) * scale(s) * translate(-srcCenter)
    glm::mat4 T(1.0f);
    T = glm::translate(T, tgtCenter);
    T = glm::scale(T, glm::vec3(s));
    T = glm::translate(T, -srcCenter);

    // Apply to all sources (vertices and normals)
    for (auto* m : sources) {
        if (!m) continue;
        for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
            glm::vec4 p(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2], 1.0f);
            p = T * p;
            m->mVertices[i]   = p.x;
            m->mVertices[i+1] = p.y;
            m->mVertices[i+2] = p.z;
        }
        // Normals unaffected by uniform positive scale + translation
    }

    std::cout << "[prealignSourceToTarget] s=" << s
              << "  (srcDiag=" << srcDiag << " -> tgtDiag=" << tgtDiag << ")"
              << "  srcCenter=(" << srcCenter.x << "," << srcCenter.y << "," << srcCenter.z << ")"
              << " tgtCenter=(" << tgtCenter.x << "," << tgtCenter.y << "," << tgtCenter.z << ")"
              << std::endl;
    return T;
}

// =============================================================================
//  Apply (or invert) a stored mat4 transform to a set of meshes.
//  Helper for rebuild paths that need to undo a previous prealignment.
// =============================================================================
inline void applyTransformToMeshes(
    const std::vector<mCutMesh*>& meshes,
    const glm::mat4& T)
{
    for (auto* m : meshes) {
        if (!m) continue;
        for (size_t i = 0; i + 2 < m->mVertices.size(); i += 3) {
            glm::vec4 p(m->mVertices[i], m->mVertices[i+1], m->mVertices[i+2], 1.0f);
            p = T * p;
            m->mVertices[i]   = p.x;
            m->mVertices[i+1] = p.y;
            m->mVertices[i+2] = p.z;
        }
    }
}

// =============================================================================
//  PHASE-A diagnostic: median nearest-neighbor distance of a point cloud.
// -----------------------------------------------------------------------------
//  Defines the local sampling resolution L of the target cloud.
//  L is the natural reference length for FPFH / FGR / ICP parameter scaling
//  (Rusu 2009, Zhou 2016, Open3D tutorials).
//
//  Implementation:
//      - For each point, find its 1st non-self nearest neighbor (k=2 then take
//        the second result, since the 1st result is the point itself with
//        distance 0).
//      - Take the median of all NN distances (robust against outliers).
//
//  This is currently used for LOGGING ONLY -- registration parameters are
//  not yet derived from L. Phase B will switch to L-based scaling.
// =============================================================================
inline float computeMedianNNDistance(const PointCloud& cloud) {
    const size_t N = cloud.points.size();
    if (N < 2) return 0.0f;

    NanoflannAdaptor adaptor(cloud.points);
    auto tree = buildKDTree(adaptor);

    std::vector<float> nnDists;
    nnDists.reserve(N);

    std::vector<size_t> indices;
    std::vector<float>  dists_sq;
    for (size_t i = 0; i < N; i++) {
        size_t k = searchKNN(*tree, cloud.points[i], 2, indices, dists_sq);
        if (k >= 2) {
            // dists_sq[0] should be ~0 (self), dists_sq[1] is the real NN
            nnDists.push_back(std::sqrt(dists_sq[1]));
        }
    }
    if (nnDists.empty()) return 0.0f;

    auto mid = nnDists.begin() + nnDists.size() / 2;
    std::nth_element(nnDists.begin(), mid, nnDists.end());
    float median = *mid;

    // Also compute mean and percentiles for diagnostic context
    std::sort(nnDists.begin(), nnDists.end());
    float p05  = nnDists[(size_t)(0.05f * nnDists.size())];
    float p95  = nnDists[(size_t)(0.95f * nnDists.size())];
    double sum = 0.0;
    for (float d : nnDists) sum += d;
    float mean = (float)(sum / nnDists.size());

    std::cout << "[L-diagnostic] median NN distance of target cloud:" << std::endl;
    std::cout << "    points         : " << N << std::endl;
    std::cout << "    median (= L)   : " << median << std::endl;
    std::cout << "    mean           : " << mean   << std::endl;
    std::cout << "    p05            : " << p05    << std::endl;
    std::cout << "    p95            : " << p95    << std::endl;
    std::cout << "    min            : " << nnDists.front() << std::endl;
    std::cout << "    max            : " << nnDists.back()  << std::endl;

    return median;
}

// =============================================================================
//  Mirror mesh and cloud along the X axis (left-right flip in image space).
// -----------------------------------------------------------------------------
//  Use this when the OBJ's 3D space is left-right flipped relative to the
//  organ CT model. Common causes:
//     - UVC camera mirror mode (webcam-style horizontal flip)
//     - Kinect color stream's "mirror" setting
//     - DICOM/CT convention mismatch (patient-right +X vs image-right +X)
//     - Pinhole unprojector using negative fx sign
//
//  IMPORTANT: Call AFTER extractTargetFromOBJ. The extraction uses the
//  original X to look up the mask; flipping before would break mask lookup.
//
//  Effects:
//     - mesh.mVertices X component negated
//     - mesh.mNormals X component negated (outward direction follows)
//     - mesh.mIndices triangle winding reversed (keeps rendering consistent
//       under back-face culling; harmless when culling is off)
//     - cloud.points X component negated
//     - cloud.normals X component negated
// =============================================================================
inline void mirrorMeshAndCloudX(mCutMesh& mesh, PointCloud& cloud) {
    // 1) flip X of mesh vertices
    for (size_t i = 0; i < mesh.mVertices.size(); i += 3) {
        mesh.mVertices[i] = -mesh.mVertices[i];
    }
    // 2) flip X of mesh normals (outward direction becomes mirror-outward)
    for (size_t i = 0; i < mesh.mNormals.size(); i += 3) {
        mesh.mNormals[i] = -mesh.mNormals[i];
    }
    // 3) reverse triangle winding so cross-product face normals stay outward
    for (size_t i = 0; i + 2 < mesh.mIndices.size(); i += 3) {
        std::swap(mesh.mIndices[i + 1], mesh.mIndices[i + 2]);
    }
    // 4) flip X of cloud points and normals
    for (auto& p : cloud.points)  p.x = -p.x;
    for (auto& n : cloud.normals) n.x = -n.x;

    std::cout << "[mirrorMeshAndCloudX] flipped X axis"
              << " (mesh verts=" << (mesh.mVertices.size() / 3)
              << ", tris=" << (mesh.mIndices.size() / 3)
              << ", cloud points=" << cloud.points.size() << ")" << std::endl;
}

}  // namespace Reg3DCustom
