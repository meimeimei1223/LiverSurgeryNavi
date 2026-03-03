#pragma once
// =============================================================================
// PlatformCompat.h
// =============================================================================

#ifdef _WIN32

#include <windows.h>
#include <sys/stat.h>
#include <direct.h>

#ifndef usleep
#define usleep(us) Sleep((us) / 1000)
#endif

#ifndef WEXITSTATUS
#define WEXITSTATUS(status) (status)
#endif

#define DEV_NULL "NUL"
#define PLATFORM_POPEN _popen
#define PLATFORM_PCLOSE _pclose

#include <string>
#include <filesystem>
inline void platform_mkdir_p(const std::string& path) {
    std::filesystem::create_directories(path);
}

#define EXE_SUFFIX ".exe"

#else

#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

#define DEV_NULL "/dev/null"
#define PLATFORM_POPEN popen
#define PLATFORM_PCLOSE pclose

#include <string>
#include <filesystem>
inline void platform_mkdir_p(const std::string& path) {
    std::filesystem::create_directories(path);
}

#define EXE_SUFFIX ""
#endif
