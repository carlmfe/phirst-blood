cmake_minimum_required(VERSION 3.18)

project(RamboKokkos 
    VERSION 1.0.0
    DESCRIPTION "RAMBO Phase Space Generator Library for Kokkos"
    LANGUAGES CXX)

# =============================================================================
# Find Kokkos FIRST (it may set required compiler settings)
# =============================================================================
# Kokkos can be found via:
#   1. CMAKE_PREFIX_PATH pointing to Kokkos installation
#   2. Kokkos_ROOT environment variable
#   3. System-wide installation
#
# Example: cmake -DKokkos_ROOT=/path/to/kokkos/install ..
# =============================================================================
find_package(Kokkos REQUIRED)

# Print Kokkos configuration
message(STATUS "Kokkos version: ${Kokkos_VERSION}")
message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Options
# =============================================================================
option(RAMBO_BUILD_EXAMPLES "Build example executables" ON)
option(RAMBO_INSTALL "Generate install target" ON)

# =============================================================================
# Header-only Library Target
# =============================================================================
add_library(rambo_kokkos_lib INTERFACE)
add_library(rambo::kokkos ALIAS rambo_kokkos_lib)

target_include_directories(rambo_kokkos_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(rambo_kokkos_lib INTERFACE Kokkos::kokkos)

# Require C++17 minimum (Kokkos requirement)
target_compile_features(rambo_kokkos_lib INTERFACE cxx_std_17)

# =============================================================================
# Example Executable
# =============================================================================
if(RAMBO_BUILD_EXAMPLES)
    add_executable(rambo_kokkos main.cpp)
    target_link_libraries(rambo_kokkos PRIVATE rambo_kokkos_lib)
    
    # Platform-specific linking
    if(UNIX AND NOT APPLE)
        target_link_libraries(rambo_kokkos PRIVATE m)
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================
if(RAMBO_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
    # Install headers
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp"
    )
    
    # Install library target
    install(TARGETS rambo_kokkos_lib
        EXPORT rambo-kokkos-targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    # Install export set
    install(EXPORT rambo-kokkos-targets
        FILE rambo-kokkos-targets.cmake
        NAMESPACE rambo::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-kokkos
    )
    
    # Generate and install package config
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rambo-kokkosConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-kokkosConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-kokkos
    )
    
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-kokkosConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-kokkosConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-kokkosConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-kokkos
    )
endif()

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
