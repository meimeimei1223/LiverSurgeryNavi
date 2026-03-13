cmake_minimum_required(VERSION 3.14)
project(medsam2_da3_lite LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA support" OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/wd4819 /wd4267 /wd4244 /wd4996 /MP /utf-8)
endif()

if(WIN32)
    set(ORT_NAME "onnxruntime-win-x64-1.15.1")
    set(ORT_ARCHIVE_EXT "zip")
else()
    set(ORT_NAME "onnxruntime-linux-x64-1.15.1")
    set(ORT_ARCHIVE_EXT "tgz")
endif()

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/${ORT_NAME}"
    CACHE PATH "ONNX Runtime installation path")

if(NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR
        "ONNX Runtime not found at: ${ONNXRUNTIME_ROOT}\n"
        "Please download from: https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1\n"
        "  Windows: onnxruntime-win-x64-1.15.1.zip -> ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-1.15.1/\n"
        "  Linux:   onnxruntime-linux-x64-1.15.1.tgz\n"
        "Or set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR     ${ONNXRUNTIME_ROOT}/lib)

set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

foreach(STB_FILE stb_image.h stb_image_write.h stb_image_resize2.h)
    if(NOT EXISTS ${STB_DIR}/${STB_FILE})
        message(STATUS "Downloading ${STB_FILE}...")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/nothings/stb/master/${STB_FILE}"
            "${STB_DIR}/${STB_FILE}"
            SHOW_PROGRESS
        )
    endif()
endforeach()

add_library(medsam2_da3_lite_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/depth_anything_v3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sam2_segmentor.cpp
)

target_include_directories(medsam2_da3_lite_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_directories(medsam2_da3_lite_lib PUBLIC ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(medsam2_da3_lite_lib PUBLIC onnxruntime)

if(USE_CUDA)
    target_compile_definitions(medsam2_da3_lite_lib PRIVATE USE_CUDA)
endif()

add_executable(medsam2_da3_lite ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(medsam2_da3_lite PRIVATE medsam2_da3_lite_lib)

if(WIN32)
    add_custom_command(TARGET medsam2_da3_lite POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:medsam2_da3_lite>"
    )
else()
    set_target_properties(medsam2_da3_lite PROPERTIES
        BUILD_RPATH  ${ONNXRUNTIME_LIB_DIR}
        INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
    )
endif()

install(TARGETS medsam2_da3_lite DESTINATION bin)

message(STATUS "")
message(STATUS "=== medsam2_da3_lite Build Configuration ===")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "CUDA Support: ${USE_CUDA}")
message(STATUS "Platform:     ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler:     ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "=============================================")
message(STATUS "")
