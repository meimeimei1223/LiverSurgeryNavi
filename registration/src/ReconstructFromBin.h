#pragma once
// =============================================================================
//  ReconstructFromBin.h  (FEATURE_PLAN_external_depth_drop.md Task 3)
//  ---------------------------------------------------------------------------
//  "Reconstruct from BIN" (経路 D): take a depth_metric.bin + segmentation_mask.png
//  (+ optional original.jpg) the user drops into the REG UI and regenerate the
//  canonical OBJ via common::exportDepthArtifacts() WITHOUT running DA3/SAM2.
//
//  This header holds the drop-slot state machine + action declarations. File
//  loading (onFileDropped/onFolderDropped) is implemented in ReconstructFromBin.cpp;
//  execute() (the Reconstruct action) is filled in Task 6.
// =============================================================================
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include "DepthToObjExport.h"      // depthexport::DepthBin
#include "image_utils.hpp"         // img::Image
#include "OBJTargetExtraction.h"   // Reg3DCustom::CameraIntrinsics

namespace ReconstructFromBin {

enum class SlotStatus { Empty, Loaded, Error };

struct Slot {
    SlotStatus  status = SlotStatus::Empty;
    std::string filePath;       // original drop-source path
    std::string errorMessage;   // set when status == Error
    int         width = 0, height = 0;  // populated when Loaded
};

struct State {
    Slot bin;    // depth_metric.bin (required)
    Slot mask;   // segmentation_mask.png (required)
    Slot rgb;    // original.jpg (optional, for texture)

    // Loaded payloads (filled on successful drop; consumed by execute()).
    depthexport::DepthBin depthData;
    std::vector<uint8_t>  maskData;   // H*W, 0/255
    img::Image            rgbImage;   // optional

    // bin + mask loaded and their resolutions agree (rgb optional).
    bool isReconstructReady() const {
        return bin.status  == SlotStatus::Loaded
            && mask.status == SlotStatus::Loaded
            && (rgb.status == SlotStatus::Loaded || rgb.status == SlotStatus::Empty)
            && bin.width  == mask.width
            && bin.height == mask.height;
    }
    bool hasAny() const {
        return bin.status  != SlotStatus::Empty
            || mask.status != SlotStatus::Empty
            || rgb.status  != SlotStatus::Empty;
    }
    void clear() {
        bin = Slot{}; mask = Slot{}; rgb = Slot{};
        depthData = depthexport::DepthBin{};
        maskData.clear();
        rgbImage = img::Image{};
    }
};

// True iff a dropped path should be routed to Reconstruct (vs the normal
// image-load path). Conservative: *.bin, or the canonical names
// segmentation_mask.png / original.jpg. A generic image keeps going to the
// normal Lap-Images drop handler.
bool isReconstructFile(const std::string& path);

// Image decoder injected by the caller. The registration app owns the single
// stb_image implementation (in main.cpp), so PNG/JPG decoding is delegated here
// rather than linked into this TU. Returns true on success; fills `out`
// (channels-interleaved, row-major), w, h. `channels` is the requested channel
// count (1 = grayscale mask, 3 = RGB).
using ImageDecoder = std::function<bool(const std::string& path, int channels,
                                        std::vector<uint8_t>& out, int& w, int& h)>;

// Load a dropped reconstruct file into the matching slot (by name/extension).
void onFileDropped(State& state, const std::string& path, const ImageDecoder& decode);

// Scan a dropped folder for depth_metric.bin / segmentation_mask.png /
// original.jpg and load whichever are present.
void onFolderDropped(State& state, const std::string& folderPath, const ImageDecoder& decode);

// Reconstruct action (implemented in Task 6).
struct ReconstructResult {
    bool        ok = false;
    std::string errorMessage;
    std::string backupPath;   // depth_output_backup_<ts>/ created before overwrite
};
ReconstructResult execute(State& state,
                          const std::string& depthOutputPath,
                          const Reg3DCustom::CameraIntrinsics& currentK);

} // namespace ReconstructFromBin
