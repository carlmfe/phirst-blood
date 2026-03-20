cmake_minimum_required(VERSION 3.18)
project("RamboAlpaka" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Backend Selection
# =============================================================================
set(ALPAKA_BACKEND "CPU" CACHE STRING "Alpaka backend: CUDA, HIP, SYCL, CPU, OMP")
set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS CUDA HIP SYCL CPU OMP)
message(STATUS "Alpaka backend: ${ALPAKA_BACKEND}")

# Configure backend-specific settings BEFORE finding alpaka
set(alpaka_RELOCATABLE_DEVICE_CODE ON)

if(ALPAKA_BACKEND STREQUAL "CUDA")
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA backend requested but no CUDA compiler found.")
    endif()
    enable_language(CUDA)
    set(alpaka_ACC_GPU_CUDA_ENABLE ON CACHE BOOL "" FORCE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
    endif()
    add_compile_definitions(ALPAKA_USE_CUDA)
elseif(ALPAKA_BACKEND STREQUAL "CPU")
    set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
    add_compile_definitions(ALPAKA_USE_CPU_SERIAL)
elseif(ALPAKA_BACKEND STREQUAL "OMP")
    find_package(OpenMP REQUIRED)
    set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
    add_compile_definitions(ALPAKA_USE_CPU_OMP)
endif()

find_package(alpaka REQUIRED)
message(STATUS "Alpaka version: ${alpaka_VERSION}")

include(GNUInstallDirs)

# =============================================================================
# Library Target (Header-Only INTERFACE Library)
# =============================================================================

add_library(rambo_alpaka_lib INTERFACE)
add_library(rambo::alpaka ALIAS rambo_alpaka_lib)

target_include_directories(rambo_alpaka_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link alpaka for build, but not for install (users must find_package(alpaka) themselves)
target_link_libraries(rambo_alpaka_lib INTERFACE $<BUILD_INTERFACE:alpaka::alpaka>)

# =============================================================================
# Example Executable (not part of library)
# =============================================================================

alpaka_add_executable(rambo_alpaka main.cpp)
target_link_libraries(rambo_alpaka PUBLIC rambo_alpaka_lib)

if(ALPAKA_BACKEND STREQUAL "OMP")
    target_link_libraries(rambo_alpaka PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# Installation
# =============================================================================

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS rambo_alpaka_lib
    EXPORT rambo-alpaka-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT rambo-alpaka-targets
    FILE rambo-alpaka-targets.cmake
    NAMESPACE rambo::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-alpaka
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rambo-alpakaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-alpakaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-alpaka
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-alpakaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-alpakaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-alpakaConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-alpaka
)
