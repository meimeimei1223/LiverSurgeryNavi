cmake_minimum_required(VERSION 3.16)
project(LiverSurgeryNaviComb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ---------------------------------------------------------------------------
# Build target selection
#   - lsn_registration : REGISTRATION app  (registration/main.cpp + reg_specific)
#   - lsn_deform       : DEFORM app         (deform/main.cpp + deform_specific)
#   Both share common/src (33 unified headers + 5 .cpp).
# ---------------------------------------------------------------------------
option(BUILD_REG    "Build REGISTRATION app (lsn_registration)" ON)
option(BUILD_DEFORM "Build DEFORM app (lsn_deform)"             ON)
option(ENABLE_AVX2  "Enable AVX2" ON)
option(ENABLE_SSE2  "Enable SSE2" ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/wd4819 /wd4005 /wd4267 /wd4244 /wd4996 /wd4458 /wd4312 /wd4305 /MP /utf-8)
    if(ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wno-volatile -Wno-sign-compare -Wno-unused-parameter -Wno-conversion -Wno-deprecated-declarations)
    if(ENABLE_AVX2)
        add_compile_options(-mavx2)
    endif()
endif()

# ===========================================================================
# Third-party detection (shared by all apps) -- preserved from original build
# ===========================================================================
if(EXISTS "${CMAKE_SOURCE_DIR}/win_deps")
    set(WIN_DEPS_DIR "${CMAKE_SOURCE_DIR}/win_deps")
else()
    set(WIN_DEPS_DIR "C:/MyGitHubProjcetWin/AAA_Reg3D_ONNX/win_deps")
endif()
set(INCLUDES_DIR "${WIN_DEPS_DIR}/includes")
set(LIBS_DIR     "${WIN_DEPS_DIR}/lib")

if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/glm")
    set(GLM_DIR "${CMAKE_SOURCE_DIR}/third_party/glm")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/glm")
    set(GLM_DIR "${CMAKE_SOURCE_DIR}/glm")
else()
    set(GLM_DIR "${INCLUDES_DIR}/glm")
endif()

set(STB_IMAGE_DIR "${CMAKE_SOURCE_DIR}/third_party/stb")

if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/imgui/imgui.h")
    set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/imgui/imgui.h")
    set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/imgui")
else()
    message(FATAL_ERROR "ImGui not found!")
endif()
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set(TINYOBJ_DIR "${CMAKE_SOURCE_DIR}/third_party/tinyobjloader")
if(EXISTS "${TINYOBJ_DIR}/tiny_obj_loader.h")
    set(TINYOBJ_SOURCES "${TINYOBJ_DIR}/tiny_obj_loader_impl.cpp")
    message(STATUS "tinyobjloader: ${TINYOBJ_DIR}")
else()
    message(FATAL_ERROR "tinyobjloader not found at ${TINYOBJ_DIR}")
endif()

# Third-party C sources compiled into every app
set(LSN_TP_SOURCES "")
if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs/tinyfiledialogs.c)
    list(APPEND LSN_TP_SOURCES ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs/tinyfiledialogs.c)
    add_compile_definitions(HAS_TINYFILEDIALOGS)
    message(STATUS "tinyfiledialogs found - file picker enabled")
else()
    message(STATUS "tinyfiledialogs not found - drag & drop only")
endif()

# c-cmaes (Apache 2.0). Prefer tls_wrapper for OpenMP Run-level parallelism.
set(CMAES_DIR "${CMAKE_SOURCE_DIR}/third_party/c-cmaes")
if(EXISTS "${CMAES_DIR}/cmaes_tls_wrapper.c" AND EXISTS "${CMAES_DIR}/cmaes.c")
    list(APPEND LSN_TP_SOURCES "${CMAES_DIR}/cmaes_tls_wrapper.c")
    message(STATUS "c-cmaes: ${CMAES_DIR} (via tls_wrapper)")
elseif(EXISTS "${CMAES_DIR}/cmaes.c")
    list(APPEND LSN_TP_SOURCES "${CMAES_DIR}/cmaes.c")
    message(WARNING "c-cmaes: cmaes_tls_wrapper.c not found, using cmaes.c directly (Run parallelism disabled)")
else()
    message(FATAL_ERROR "c-cmaes not found at ${CMAES_DIR}")
endif()

# Eigen include dir
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/eigen/Eigen/Dense")
    set(EIGEN_INC "${CMAKE_SOURCE_DIR}/third_party/eigen")
elseif(EXISTS "${INCLUDES_DIR}/eigen3/Eigen/Dense")
    set(EIGEN_INC "${INCLUDES_DIR}/eigen3")
elseif(EXISTS "${INCLUDES_DIR}/Eigen/Dense")
    set(EIGEN_INC "${INCLUDES_DIR}")
else()
    set(EIGEN_INC "")
    message(WARNING "Eigen not found in third_party/ or win_deps; relying on system include")
endif()

# Common include dirs shared by every app (third_party + common/src)
set(LSN_TP_INCLUDES
    ${CMAKE_SOURCE_DIR}/common/src
    ${STB_IMAGE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs
    ${CMAKE_SOURCE_DIR}/third_party/nanoflann
    ${TINYOBJ_DIR}
    ${GLM_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAES_DIR}
)
if(EIGEN_INC)
    list(APPEND LSN_TP_INCLUDES ${EIGEN_INC})
endif()
if(WIN32)
    list(APPEND LSN_TP_INCLUDES ${INCLUDES_DIR})
endif()

# ===========================================================================
# Platform libraries
# ===========================================================================
find_package(OpenMP)
if(NOT WIN32)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glfw3 REQUIRED)
endif()

# ===========================================================================
# Sub-tools (built once, shared by both apps)
# ===========================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

if(EXISTS ${CMAKE_SOURCE_DIR}/sam2_da3_lite/CMakeLists.txt)
    message(STATUS "=== Building sam2_da3_lite ===")
    add_subdirectory(${CMAKE_SOURCE_DIR}/sam2_da3_lite ${PROJECT_BINARY_DIR}/sam2_da3_lite)
    set_target_properties(sam2_da3_lite PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
else()
    message(WARNING "sam2_da3_lite/ not found - depth estimation will use fallback")
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/calibration_tool/CMakeLists.txt)
    message(STATUS "=== Building calibration_tool ===")
    add_subdirectory(${CMAKE_SOURCE_DIR}/calibration_tool ${PROJECT_BINARY_DIR}/calibration_tool)
    set_target_properties(calibration_tool PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
else()
    message(STATUS "calibration_tool/ not found - camera calibration disabled")
endif()

# ===========================================================================
# Common .cpp sources (compiled into each app)
# Headers are globbed too so Qt Creator / IDEs list them in the project tree.
# (CMake ignores .h/.hpp for compilation; they only populate the tree.)
# ===========================================================================
file(GLOB LSN_COMMON_CPP CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/common/src/*.cpp)
file(GLOB LSN_COMMON_HDR CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/common/src/*.h
    ${CMAKE_SOURCE_DIR}/common/src/*.hpp)

# ---------------------------------------------------------------------------
# Helper: apply shared configuration to an app target.
#   tgt          : target name
#   app_src_dir  : the app-specific src dir (registration/src or deform/src)
#   mode_define  : extra compile definition (LSN_MODE_REGISTRATION / _DEFORM)
# ---------------------------------------------------------------------------
function(lsn_configure_app tgt app_src_dir mode_define)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${app_src_dir}
        ${LSN_TP_INCLUDES}
    )
    if(NOT WIN32)
        target_include_directories(${tgt} PRIVATE
            ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR})
    endif()

    target_compile_definitions(${tgt} PRIVATE
        GLEW_STATIC _USE_MATH_DEFINES NOMINMAX ${mode_define})

    if(WIN32)
        target_link_directories(${tgt} PRIVATE ${LIBS_DIR})
        target_link_libraries(${tgt} PRIVATE opengl32 glew32s glfw3_mt)
    else()
        target_link_libraries(${tgt} PRIVATE ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(OpenMP_C_FOUND)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_C)
    endif()

    if(MSVC)
        target_compile_options(${tgt} PRIVATE
            $<$<CONFIG:Release>:/O2 /Ob2 /Oi /Ot /GL>
            $<$<CONFIG:Debug>:/Od /Zi>)
        target_link_options(${tgt} PRIVATE $<$<CONFIG:Release>:/LTCG>)
    else()
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${tgt} PRIVATE -O3 -march=native -DNDEBUG)
        else()
            target_compile_options(${tgt} PRIVATE -O1 -g)
        endif()
        if(OpenMP_CXX_FOUND)
            target_compile_options(${tgt} PRIVATE -fopenmp)
        endif()
    endif()

    if(WIN32)
        set_target_properties(${tgt} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_BINARY_DIR}/bin
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
    endif()
    set_target_properties(${tgt} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endfunction()

# ===========================================================================
# REGISTRATION app
# ===========================================================================
if(BUILD_REG)
    file(GLOB LSN_REG_CPP CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/registration/src/*.cpp)
    file(GLOB LSN_REG_HDR CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/registration/src/*.h
        ${CMAKE_SOURCE_DIR}/registration/src/*.hpp)
    add_executable(lsn_registration
        ${CMAKE_SOURCE_DIR}/registration/main.cpp
        ${LSN_REG_CPP}
        ${LSN_REG_HDR}
        ${LSN_COMMON_CPP}
        ${LSN_COMMON_HDR}
        ${IMGUI_SOURCES}
        ${TINYOBJ_SOURCES}
        ${LSN_TP_SOURCES}
    )
    lsn_configure_app(lsn_registration ${CMAKE_SOURCE_DIR}/registration/src LSN_MODE_REGISTRATION)
endif()

# ===========================================================================
# DEFORM app
# ===========================================================================
if(BUILD_DEFORM)
    file(GLOB LSN_DEF_CPP CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/deform/src/*.cpp)
    file(GLOB LSN_DEF_HDR CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/deform/src/*.h
        ${CMAKE_SOURCE_DIR}/deform/src/*.hpp)
    add_executable(lsn_deform
        ${CMAKE_SOURCE_DIR}/deform/main.cpp
        ${LSN_DEF_CPP}
        ${LSN_DEF_HDR}
        ${LSN_COMMON_CPP}
        ${LSN_COMMON_HDR}
        ${IMGUI_SOURCES}
        ${TINYOBJ_SOURCES}
        ${LSN_TP_SOURCES}
    )
    lsn_configure_app(lsn_deform ${CMAKE_SOURCE_DIR}/deform/src LSN_MODE_DEFORM)
endif()

# ---------------------------------------------------------------------------
# The apps spawn external helper executables (PathConfig.h):
#   - sam2_da3_lite  : depth estimation / segmentation (DEPTH_EXE_PATH)
#   - calibration_tool : chessboard camera calibration
# Make the apps depend on them so building/running an app in an IDE (Qt Creator
# builds only the active target) also builds the helpers into bin/, otherwise
# they resolve to NOT FOUND at runtime.
# ---------------------------------------------------------------------------
foreach(app lsn_registration lsn_deform)
    if(TARGET ${app})
        foreach(helper sam2_da3_lite calibration_tool)
            if(TARGET ${helper})
                add_dependencies(${app} ${helper})
            endif()
        endforeach()
    endif()
endforeach()

# ===========================================================================
# Shared runtime resources -> bin/  (single shared bin dir)
# ===========================================================================
foreach(res shaders data model registration_model input_image)
    if(EXISTS ${CMAKE_SOURCE_DIR}/${res})
        file(COPY ${CMAKE_SOURCE_DIR}/${res} DESTINATION ${CMAKE_BINARY_DIR}/bin)
    endif()
endforeach()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Project:        ${PROJECT_NAME}")
message(STATUS "BUILD_REG:      ${BUILD_REG}  -> lsn_registration")
message(STATUS "BUILD_DEFORM:   ${BUILD_DEFORM}  -> lsn_deform")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform:       ${CMAKE_SYSTEM_NAME}")
message(STATUS "GLM:            ${GLM_DIR}")
message(STATUS "ImGui:          ${IMGUI_DIR}")
message(STATUS "Eigen:          ${EIGEN_INC}")
message(STATUS "OpenMP:         ${OpenMP_CXX_FOUND}")
message(STATUS "Output:         ${PROJECT_BINARY_DIR}/bin/")
message(STATUS "========================================")
message(STATUS "")
