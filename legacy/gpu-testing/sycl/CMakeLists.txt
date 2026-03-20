cmake_minimum_required(VERSION 3.18)

# =============================================================================
# SYCL Backend Selection
# =============================================================================
set(SYCL_BACKEND "CUDA" CACHE STRING "SYCL backend: CUDA, HIP, INTEL, ADAPTIVECPP")
set_property(CACHE SYCL_BACKEND PROPERTY STRINGS CUDA HIP INTEL ADAPTIVECPP)

# GPU Architecture
set(CUDA_GPU_ARCH "sm_89" CACHE STRING "NVIDIA GPU architecture")
set(HIP_GPU_ARCH "gfx906" CACHE STRING "AMD GPU architecture")

project("RamboSYCL" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

# =============================================================================
# Library Target (Header-Only INTERFACE Library)
# =============================================================================

add_library(rambo_sycl_lib INTERFACE)
add_library(rambo::sycl ALIAS rambo_sycl_lib)

target_include_directories(rambo_sycl_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# =============================================================================
# Example Executable (not part of library)
# =============================================================================

add_executable(rambo_sycl main.cpp)
target_link_libraries(rambo_sycl PRIVATE rambo_sycl_lib)

# Backend-specific flags
if(SYCL_BACKEND STREQUAL "CUDA")
    target_compile_options(rambo_sycl PRIVATE 
        -fsycl
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend 
        --cuda-gpu-arch=${CUDA_GPU_ARCH}
        $<$<CONFIG:Release>:-O3>
    )
    target_link_options(rambo_sycl PRIVATE
        -fsycl
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend 
        --cuda-gpu-arch=${CUDA_GPU_ARCH}
    )
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_CUDA)
elseif(SYCL_BACKEND STREQUAL "HIP")
    target_compile_options(rambo_sycl PRIVATE 
        -fsycl
        -fsycl-targets=amdgcn-amd-amdhsa
        -Xsycl-target-backend
        --offload-arch=${HIP_GPU_ARCH}
        $<$<CONFIG:Release>:-O3>
    )
    target_link_options(rambo_sycl PRIVATE
        -fsycl
        -fsycl-targets=amdgcn-amd-amdhsa
        -Xsycl-target-backend
        --offload-arch=${HIP_GPU_ARCH}
    )
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_HIP)
elseif(SYCL_BACKEND STREQUAL "INTEL")
    target_compile_options(rambo_sycl PRIVATE -fsycl $<$<CONFIG:Release>:-O3>)
    target_link_options(rambo_sycl PRIVATE -fsycl)
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_INTEL)
elseif(SYCL_BACKEND STREQUAL "ADAPTIVECPP")
    find_package(AdaptiveCpp REQUIRED)
    add_sycl_to_target(TARGET rambo_sycl)
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_ADAPTIVECPP)
endif()

# =============================================================================
# Installation
# =============================================================================

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS rambo_sycl_lib
    EXPORT rambo-sycl-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT rambo-sycl-targets
    FILE rambo-sycl-targets.cmake
    NAMESPACE rambo::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-sycl
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rambo-syclConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-syclConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-sycl
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-syclConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-syclConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-syclConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-sycl
)
