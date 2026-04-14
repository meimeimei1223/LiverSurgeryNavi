cmake_minimum_required(VERSION 3.16)
project(LiverSurgeryNavi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(ENABLE_AVX2 "Enable AVX2" ON)
option(ENABLE_SSE2 "Enable SSE2" ON)

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

if(EXISTS ${CMAKE_SOURCE_DIR}/medsam2_da3_lite/CMakeLists.txt)
    message(STATUS "")
    message(STATUS "=== Building medsam2_da3_lite ===")
    add_subdirectory(${CMAKE_SOURCE_DIR}/medsam2_da3_lite ${PROJECT_BINARY_DIR}/medsam2_da3_lite)
    set_target_properties(medsam2_da3_lite PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    )
    message(STATUS "medsam2_da3_lite -> ${PROJECT_BINARY_DIR}/bin/")
    message(STATUS "=================================")
    message(STATUS "")
else()
    message(WARNING "medsam2_da3_lite/ not found - depth estimation will use fallback")
endif()

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

if(EXISTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
    file(GLOB SOURCES
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.h
        ${CMAKE_SOURCE_DIR}/src/*.hpp
        ${CMAKE_SOURCE_DIR}/src/*.glsl
    )
else()
    file(GLOB SOURCES
        ${CMAKE_SOURCE_DIR}/*.cpp
        ${CMAKE_SOURCE_DIR}/*.h
        ${CMAKE_SOURCE_DIR}/*.hpp
        ${CMAKE_SOURCE_DIR}/*.glsl
    )
    list(FILTER SOURCES EXCLUDE REGEX ".*/depth_anything_v3\\.cpp$")
    list(FILTER SOURCES EXCLUDE REGEX ".*/image_utils\\.cpp$")
    list(FILTER SOURCES EXCLUDE REGEX ".*/sam2_segmentor\\.cpp$")
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs/tinyfiledialogs.c)
    list(APPEND SOURCES ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs/tinyfiledialogs.c)
    add_compile_definitions(HAS_TINYFILEDIALOGS)
    message(STATUS "tinyfiledialogs found - file picker enabled")
else()
    message(STATUS "tinyfiledialogs not found - drag & drop only")
endif()

# c-cmaes (Apache 2.0, Nikolaus Hansen 2014)
set(CMAES_DIR "${CMAKE_SOURCE_DIR}/third_party/c-cmaes")
if(EXISTS "${CMAES_DIR}/cmaes.c")
    list(APPEND SOURCES "${CMAES_DIR}/cmaes.c")
    message(STATUS "c-cmaes: ${CMAES_DIR}")
else()
    message(FATAL_ERROR "c-cmaes not found at ${CMAES_DIR}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${IMGUI_SOURCES}
    ${TINYOBJ_SOURCES}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLEW_STATIC
    _USE_MATH_DEFINES
    NOMINMAX
)

if(WIN32)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${STB_IMAGE_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs
        ${CMAKE_SOURCE_DIR}/third_party/nanoflann
        ${TINYOBJ_DIR}
        ${INCLUDES_DIR}
        ${GLM_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${CMAES_DIR}
    )
    target_link_directories(${PROJECT_NAME} PRIVATE ${LIBS_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE opengl32 glew32s glfw3_mt)
else()
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glfw3 REQUIRED)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${STB_IMAGE_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/tinyfiledialogs
        ${CMAKE_SOURCE_DIR}/third_party/nanoflann
        ${TINYOBJ_DIR}
        ${GLM_DIR}
        ${GLEW_INCLUDE_DIRS}
        ${GLFW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${CMAES_DIR}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
        glfw
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/eigen/Eigen/Dense")
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/third_party/eigen)
    message(STATUS "Eigen3: ${CMAKE_SOURCE_DIR}/third_party/eigen")
elseif(EXISTS "${INCLUDES_DIR}/eigen3/Eigen/Dense")
    target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDES_DIR}/eigen3)
    message(STATUS "Eigen3: ${INCLUDES_DIR}/eigen3")
elseif(EXISTS "${INCLUDES_DIR}/Eigen/Dense")
    target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDES_DIR})
    message(STATUS "Eigen3: ${INCLUDES_DIR}")
else()
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(Eigen3_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
    else()
        message(FATAL_ERROR "Eigen3 not found!")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP: ON (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "OpenMP: OFF")
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:/O2 /Ob2 /Oi /Ot /GL>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/LTCG>)
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native -DNDEBUG)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -O1 -g)
    endif()
    if(OpenMP_CXX_FOUND)
        target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
    endif()
endif()

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${PROJECT_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin
    )
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/shaders)
    file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/data)
    file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()
if(EXISTS ${CMAKE_SOURCE_DIR}/model)
    file(COPY ${CMAKE_SOURCE_DIR}/model DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Project:    ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Platform:   ${CMAKE_SYSTEM_NAME}")
message(STATUS "win_deps:   ${WIN_DEPS_DIR}")
message(STATUS "GLM:        ${GLM_DIR}")
message(STATUS "ImGui:      ${IMGUI_DIR}")
message(STATUS "stb_image:  ${STB_IMAGE_DIR}")
message(STATUS "tinyobj:    ${TINYOBJ_DIR}")
message(STATUS "OpenMP:     ${OpenMP_CXX_FOUND}")
message(STATUS "Output:     ${PROJECT_BINARY_DIR}/bin/")
message(STATUS "========================================")
message(STATUS "")
