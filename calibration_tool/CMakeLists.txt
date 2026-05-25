cmake_minimum_required(VERSION 3.16)
project(calibration_tool LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/wd4819 /wd4267 /wd4244 /wd4996 /MP /utf-8)
endif()

# stb_image — try parent's third_party/stb first, then sam2's include/
set(STB_DIR "")
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/stb/stb_image.h")
    set(STB_DIR "${CMAKE_SOURCE_DIR}/third_party/stb")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stb/stb_image.h")
    set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stb")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../sam2_da3_lite/include/stb_image.h")
    set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sam2_da3_lite/include")
else()
    message(FATAL_ERROR "stb_image.h not found")
endif()

# Eigen3
set(EIGEN_DIR "")
foreach(d
    "${CMAKE_SOURCE_DIR}/third_party/eigen"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/eigen")
    if(EXISTS "${d}/Eigen/Dense")
        set(EIGEN_DIR "${d}")
        break()
    endif()
endforeach()

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ChessboardCalibration.h
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${STB_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES)

if(EIGEN_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN_DIR})
else()
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:/O2 /Ob2 /GL>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -O2)
endif()

message(STATUS "")
message(STATUS "=== calibration_tool ===")
message(STATUS "stb:   ${STB_DIR}")
message(STATUS "Eigen: ${EIGEN_DIR}")
message(STATUS "========================")
message(STATUS "")
