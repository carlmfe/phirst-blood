cmake_minimum_required(VERSION 3.18)
project("RAMBO" CXX)

# =============================================================================
# Library Configuration
# =============================================================================

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable Kokkos backend
option(WITH_KOKKOS "Build RAMBO with Kokkos backend" OFF)

# Determine library variant and source directory
if(WITH_KOKKOS)
    set(RAMBO_VARIANT "kokkos")
    set(RAMBO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../gpu-testing/kokkos")
    message(STATUS "Building RAMBO library with Kokkos backend")
else()
    set(RAMBO_VARIANT "base")
    set(RAMBO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../gpu-testing/base")
    message(STATUS "Building RAMBO library with serial base backend")
endif()

# Verify source directory exists
if(NOT EXISTS "${RAMBO_SOURCE_DIR}/include/rambo")
    message(FATAL_ERROR "RAMBO source directory not found: ${RAMBO_SOURCE_DIR}/include/rambo")
endif()

# =============================================================================
# Header-Only Library Setup
# =============================================================================

add_library(rambo INTERFACE)
add_library(rambo::rambo ALIAS rambo)

# Include directories with generator expressions for build and install
target_include_directories(rambo INTERFACE
    $<BUILD_INTERFACE:${RAMBO_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# C++ standard requirement
target_compile_features(rambo INTERFACE cxx_std_17)

# =============================================================================
# Backend-Specific Configuration
# =============================================================================

if(WITH_KOKKOS)
    # Find and link Kokkos
    find_package(Kokkos REQUIRED)
    target_link_libraries(rambo INTERFACE Kokkos::kokkos)
    message(STATUS "Linked Kokkos backend: ${Kokkos_VERSION}")
else()
    # Base implementation has no external dependencies
    message(STATUS "Using serial CPU implementation (no external dependencies)")
endif()

# =============================================================================
# Installation
# =============================================================================

# Install header files from the appropriate backend source
install(DIRECTORY ${RAMBO_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install and export targets for find_package()
install(TARGETS rambo
    EXPORT rambo-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT rambo-targets
    FILE rambo-targets.cmake
    NAMESPACE rambo::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo
)

# Generate and install package config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ramboConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ramboConfig.cmake
    @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ramboConfigVersion.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ramboConfigVersion.cmake
    @ONLY
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ramboConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ramboConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "RAMBO Library Configuration Summary:")
message(STATUS "  Backend Variant ......... ${RAMBO_VARIANT}")
message(STATUS "  Source Directory ....... ${RAMBO_SOURCE_DIR}")
message(STATUS "  Install Prefix ......... ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Include Directory ...... ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "")
