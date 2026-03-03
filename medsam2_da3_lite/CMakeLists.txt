cmake_minimum_required(VERSION 3.14)
project(medsam2_da3_lite LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# オプション
# =============================================================================
option(USE_CUDA "Enable CUDA support" OFF)

# =============================================================================
# MSVC 設定
# =============================================================================
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/wd4819 /wd4267 /wd4244 /wd4996)
    add_compile_options(/MP /utf-8)
endif()

# =============================================================================
# ONNX Runtime（プラットフォーム分岐）
# =============================================================================
if(WIN32)
    set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.15.1"
        CACHE PATH "ONNX Runtime installation path")
else()
    set(ONNXRUNTIME_ROOT "/home/meidaikasai/onnxruntime-linux-x64-1.15.1"
        CACHE PATH "ONNX Runtime installation path")
endif()

if(NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR
        "ONNX Runtime not found at: ${ONNXRUNTIME_ROOT}\n"
        "Please download from: https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1\n"
        "  Windows: onnxruntime-win-x64-1.15.1.zip -> ${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.15.1/\n"
        "  Linux:   onnxruntime-linux-x64-1.15.1.tgz\n"
        "Or set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR     ${ONNXRUNTIME_ROOT}/lib)

# =============================================================================
# stb_image（ダウンロードまたは自動取得）
# =============================================================================
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT EXISTS ${STB_DIR}/stb_image.h)
    message(STATUS "Downloading stb_image.h...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
        "${STB_DIR}/stb_image.h"
        SHOW_PROGRESS
    )
endif()

if(NOT EXISTS ${STB_DIR}/stb_image_write.h)
    message(STATUS "Downloading stb_image_write.h...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
        "${STB_DIR}/stb_image_write.h"
        SHOW_PROGRESS
    )
endif()

if(NOT EXISTS ${STB_DIR}/stb_image_resize2.h)
    message(STATUS "Downloading stb_image_resize2.h...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h"
        "${STB_DIR}/stb_image_resize2.h"
        SHOW_PROGRESS
    )
endif()

# =============================================================================
# インクルードディレクトリ
# =============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

link_directories(${ONNXRUNTIME_LIB_DIR})

# =============================================================================
# ライブラリ
# =============================================================================
add_library(medsam2_da3_lite_lib STATIC
    src/image_utils.cpp
    src/depth_anything_v3.cpp
    src/sam2_segmentor.cpp
)

target_link_libraries(medsam2_da3_lite_lib PRIVATE onnxruntime)

if(USE_CUDA)
    target_compile_definitions(medsam2_da3_lite_lib PRIVATE USE_CUDA)
endif()

# =============================================================================
# 実行ファイル
# =============================================================================
add_executable(medsam2_da3_lite src/main.cpp)
target_link_libraries(medsam2_da3_lite PRIVATE medsam2_da3_lite_lib)

# =============================================================================
# RPATH / DLLコピー
# =============================================================================
if(WIN32)
    add_custom_command(TARGET medsam2_da3_lite POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:medsam2_da3_lite>"
        COMMENT "Copying onnxruntime.dll to output directory"
    )
else()
    set_target_properties(medsam2_da3_lite PROPERTIES
        BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
        INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
    )
endif()

# =============================================================================
# インストール
# =============================================================================
install(TARGETS medsam2_da3_lite DESTINATION bin)

# =============================================================================
# ビルド情報
# =============================================================================
message(STATUS "")
message(STATUS "=== medsam2_da3_lite Build Configuration ===")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "CUDA Support: ${USE_CUDA}")
message(STATUS "Platform:     ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler:     ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "=============================================")
message(STATUS "")
