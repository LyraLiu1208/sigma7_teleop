cmake_minimum_required(VERSION 3.16)
project(sigma7_pose_udp_sender LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SIGMA7_SDK_ROOT "" CACHE PATH "Path to the Force Dimension / Sigma7 SDK root")

if(NOT SIGMA7_SDK_ROOT)
  if(DEFINED ENV{SIGMA7_SDK_ROOT} AND NOT "$ENV{SIGMA7_SDK_ROOT}" STREQUAL "")
    set(SIGMA7_SDK_ROOT "$ENV{SIGMA7_SDK_ROOT}")
  elseif(DEFINED ENV{FORCEDIM_SDK_ROOT} AND NOT "$ENV{FORCEDIM_SDK_ROOT}" STREQUAL "")
    set(SIGMA7_SDK_ROOT "$ENV{FORCEDIM_SDK_ROOT}")
  endif()
endif()

if(NOT SIGMA7_SDK_ROOT)
  message(FATAL_ERROR "Set SIGMA7_SDK_ROOT (or FORCEDIM_SDK_ROOT) to your Linux Force Dimension SDK.")
endif()

find_path(SIGMA7_INCLUDE_DIR
  NAMES dhdc.h
  PATHS
    "${SIGMA7_SDK_ROOT}"
  PATH_SUFFIXES
    include
)

find_library(SIGMA7_DHD_LIB
  NAMES dhd libdhd
  PATHS
    "${SIGMA7_SDK_ROOT}"
  PATH_SUFFIXES
    lib
    lib64
    lib/release
    lib/release/lin-x86_64-gcc
    lib/release/lin-x86_64-clang
    lib/release/linux-gcc
    lib/release/linux-clang
    lib/release/mac-universal-clang
)

find_library(SIGMA7_DRD_LIB
  NAMES drd libdrd
  PATHS
    "${SIGMA7_SDK_ROOT}"
  PATH_SUFFIXES
    lib
    lib64
    lib/release
    lib/release/lin-x86_64-gcc
    lib/release/lin-x86_64-clang
    lib/release/linux-gcc
    lib/release/linux-clang
    lib/release/mac-universal-clang
)

if(NOT SIGMA7_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find dhdc.h under SIGMA7_SDK_ROOT=${SIGMA7_SDK_ROOT}")
endif()

if(NOT SIGMA7_DHD_LIB OR NOT SIGMA7_DRD_LIB)
  message(FATAL_ERROR "Could not find libdhd/libdrd under SIGMA7_SDK_ROOT=${SIGMA7_SDK_ROOT}")
endif()

add_executable(sigma7_pose_udp_sender ../sigma7_pose_udp_sender.cpp)
target_include_directories(sigma7_pose_udp_sender PRIVATE "${SIGMA7_INCLUDE_DIR}")
target_link_libraries(sigma7_pose_udp_sender PRIVATE "${SIGMA7_DHD_LIB}" "${SIGMA7_DRD_LIB}")

if(APPLE)
  find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
  find_library(IOKIT_FRAMEWORK IOKit)
  target_link_libraries(sigma7_pose_udp_sender PRIVATE "${COREFOUNDATION_FRAMEWORK}" "${IOKIT_FRAMEWORK}")
endif()

get_filename_component(SIGMA7_LIB_DIR "${SIGMA7_DHD_LIB}" DIRECTORY)
set_target_properties(sigma7_pose_udp_sender PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "${SIGMA7_LIB_DIR}"
  INSTALL_RPATH "${SIGMA7_LIB_DIR}"
)
