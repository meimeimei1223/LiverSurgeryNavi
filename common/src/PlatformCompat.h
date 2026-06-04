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

// =============================================================================
// Sibling-process launch (detached). Used by REG -> DEFORM and DEFORM -> REG
// transition buttons. Returns std::system's return code (informational only;
// the caller typically closes its own GLFW window right after).
// =============================================================================
#include <cstdlib>

inline int platform_launch_detached(const std::string& exePath) {
#ifdef _WIN32
    // start "" /B = empty title placeholder + no new console window.
    // The empty "" prevents exePath from being parsed as the window title.
    std::string cmd = "start \"\" /B \"" + exePath + "\"";
#else
    // Full detach: nohup (ignore SIGHUP when parent exits) + redirect all
    // standard streams to /dev/null so the child does not inherit the parent's
    // stdout/stderr pipe. Without this, when the parent exits (e.g. Qt Creator
    // closes the captured pipe), the child crashes with SIGPIPE the next time
    // it writes to stdout. The trailing & still detaches as a background job.
    std::string cmd = "nohup " + exePath + " </dev/null >/dev/null 2>&1 &";
#endif
    return std::system(cmd.c_str());
}
