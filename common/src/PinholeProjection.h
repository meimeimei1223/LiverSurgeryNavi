#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace Pinhole {

struct Intrinsics {
    float fx = 0.0f, fy = 0.0f;
    float cx = 0.0f, cy = 0.0f;
    int   width = 0, height = 0;
    bool  valid = false;
};

inline bool loadIntrinsicsTxt(const std::string& path, Intrinsics& out) {
    out = Intrinsics{};
    if (!std::filesystem::exists(path)) {
        std::cerr << "[Pinhole] intrinsics file not found: " << path << std::endl;
        return false;
    }
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[Pinhole] Cannot open: " << path << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "fx") iss >> out.fx;
        else if (key == "fy") iss >> out.fy;
        else if (key == "cx") iss >> out.cx;
        else if (key == "cy") iss >> out.cy;
        else if (key == "width")  iss >> out.width;
        else if (key == "height") iss >> out.height;
    }
    out.valid = (out.fx > 0.0f && out.fy > 0.0f &&
                 out.width > 0 && out.height > 0);
    if (out.valid) {
        std::cout << "[Pinhole] Loaded intrinsics: fx=" << out.fx
                  << " fy=" << out.fy << " cx=" << out.cx << " cy=" << out.cy
                  << " img=" << out.width << "x" << out.height << std::endl;
    } else {
        std::cerr << "[Pinhole] Invalid intrinsics in " << path << std::endl;
    }
    return out.valid;
}

}
