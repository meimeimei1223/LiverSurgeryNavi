#pragma once
#include <string>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX          // ← min/max マクロを無効化
    #endif
    #include <windows.h>

    inline std::wstring toWideString(const std::string& str) {
        if (str.empty()) return L"";
        int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
        if (size <= 0) return std::wstring(str.begin(), str.end());
        std::wstring wstr(size - 1, 0);
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size);
        return wstr;
    }

    #define ORT_MODEL_PATH(path) toWideString(path).c_str()
#else
    #define ORT_MODEL_PATH(path) (path).c_str()
#endif