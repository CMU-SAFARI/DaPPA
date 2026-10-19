cmake_minimum_required(VERSION 3.13)
project(DaPPA VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DAPPA_BUILD_TESTS "Build the DaPPA test suite" ON)
option(DAPPA_BUILD_EXAMPLES "Build the DaPPA examples" ON)

# UPMEM SDK: use $UPMEM_HOME if set, otherwise the system-wide install
# (e.g., the one in docker/Dockerfile).
if(DEFINED ENV{UPMEM_HOME})
    set(UPMEM_HOME "$ENV{UPMEM_HOME}")
    set(UPMEM_INCLUDE_DIRS "${UPMEM_HOME}/include" "${UPMEM_HOME}/include/dpu")
    set(UPMEM_LIBRARY_DIRS "${UPMEM_HOME}/lib")
else()
    set(UPMEM_INCLUDE_DIRS /usr/include/dpu)
    set(UPMEM_LIBRARY_DIRS "")
endif()

add_library(dappa SHARED
        src/arg.cpp
        src/arg_flags.cpp
        src/device.cpp
        src/kernel.cpp
        src/kernel_arg.cpp
        src/pipeline.cpp
        src/pipeline_full.cpp)

target_include_directories(dappa
        PUBLIC  include third_party src ${UPMEM_INCLUDE_DIRS})
target_link_directories(dappa PUBLIC ${UPMEM_LIBRARY_DIRS})
target_link_libraries(dappa PUBLIC dpu pthread stdc++fs)

# The DPU code template and the headers shared with the DPU are embedded into
# libdappa at compile time (see INCTXT in src/pipeline.cpp). Rebuild
# pipeline.cpp whenever one of them changes.
set_source_files_properties(src/pipeline.cpp PROPERTIES
        COMPILE_DEFINITIONS DAPPA_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        OBJECT_DEPENDS "${PROJECT_SOURCE_DIR}/src/device_template.c.inja;${PROJECT_SOURCE_DIR}/include/logging.h;${PROJECT_SOURCE_DIR}/include/shared.h")

if(DAPPA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DAPPA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
