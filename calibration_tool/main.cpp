// calibration_tool — Zhang's camera calibration from chessboard images
// Usage: calibration_tool <folder> [options]
//   --board <cols,rows>      Inner corner count (default: 9,6)
//   --square <mm>            Square size in mm (default: 22)
//   --output <path>          Output file path (default: intrinsics_calib.txt)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "ChessboardCalibration.h"

#include <iostream>
#include <string>
#include <sstream>

struct Args {
    std::string folder;
    std::string output = "intrinsics_calib.txt";
    int cols   = 9;
    int rows   = 6;
    float sqMM = 22.0f;
    bool help  = false;
};

static void printUsage(const char* name) {
    std::cout << "Usage: " << name << " <chessboard_folder> [options]\n\n"
              << "Options:\n"
              << "  --board <cols,rows>   Inner corners (default: 9,6)\n"
              << "  --square <mm>         Square size in mm (default: 22)\n"
              << "  --output <path>       Output file (default: intrinsics_calib.txt)\n"
              << "  --help                Show this help\n";
}

static Args parseArgs(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--board" && i+1 < argc) {
            std::string s = argv[++i];
            auto comma = s.find(',');
            if (comma != std::string::npos) {
                a.cols = std::stoi(s.substr(0, comma));
                a.rows = std::stoi(s.substr(comma+1));
            }
        } else if (arg == "--square" && i+1 < argc) {
            a.sqMM = std::stof(argv[++i]);
        } else if (arg == "--output" && i+1 < argc) {
            a.output = argv[++i];
        } else if (arg[0] != '-' && a.folder.empty()) {
            a.folder = arg;
        }
    }
    return a;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================\n"
              << "Chessboard Camera Calibration (Zhang)\n"
              << "========================================\n\n";

    auto args = parseArgs(argc, argv);

    if (args.help || args.folder.empty()) {
        printUsage(argv[0]);
        return args.help ? 0 : 1;
    }

    std::cout << "Folder: " << args.folder << "\n"
              << "Board:  " << args.cols << "x" << args.rows
              << " inner corners\n"
              << "Square: " << args.sqMM << " mm\n"
              << "Output: " << args.output << "\n\n";

    calib::Board board{args.cols, args.rows, args.sqMM};
    auto result = calib::calibrateFromFolder(args.folder, board);

    if (!result.valid) {
        std::cerr << "[FAIL] " << result.message << std::endl;
        return 1;
    }

    calib::saveResult(args.output, result);

    // Machine-readable summary line (parsed by main app)
    std::cout << "[CALIB_RESULT] "
              << "fx=" << result.fx << " "
              << "fy=" << result.fy << " "
              << "cx=" << result.cx << " "
              << "cy=" << result.cy << " "
              << "k1=" << result.k1 << " "
              << "k2=" << result.k2 << " "
              << "rms=" << result.rmsError << " "
              << "images=" << result.numImages << " "
              << "width=" << result.width << " "
              << "height=" << result.height << std::endl;

    return 0;
}
