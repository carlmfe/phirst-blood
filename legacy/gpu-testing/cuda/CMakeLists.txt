cmake_minimum_required(VERSION 3.18)

project("RamboCUDA" VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture
set(CMAKE_CUDA_ARCHITECTURES "native")

include(GNUInstallDirs)

# =============================================================================
# Library Target (Header-Only INTERFACE Library)
# =============================================================================

add_library(rambo_cuda_lib INTERFACE)
add_library(rambo::cuda ALIAS rambo_cuda_lib)

target_include_directories(rambo_cuda_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# =============================================================================
# Example Executable (not part of library)
# =============================================================================

add_executable(rambo_cuda main.cu)
target_link_libraries(rambo_cuda PRIVATE rambo_cuda_lib)
target_include_directories(rambo_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(rambo_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =============================================================================
# Installation
# =============================================================================

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.cuh"
)

install(TARGETS rambo_cuda_lib
    EXPORT rambo-cuda-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT rambo-cuda-targets
    FILE rambo-cuda-targets.cmake
    NAMESPACE rambo::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-cuda
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rambo-cudaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-cudaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-cuda
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-cudaConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-cudaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rambo-cudaConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-cuda
)
