cmake_minimum_required(VERSION 3.18)

project(RamboBase 
    VERSION 1.0.0
    DESCRIPTION "RAMBO Phase Space Generator Library (Serial CPU)"
    LANGUAGES CXX)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

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
add_library(rambo_base_lib INTERFACE)
add_library(rambo::base ALIAS rambo_base_lib)

target_include_directories(rambo_base_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(rambo_base_lib INTERFACE cxx_std_17)

# =============================================================================
# Example Executable
# =============================================================================
if(RAMBO_BUILD_EXAMPLES)
    add_executable(rambo_base main.cpp)
    target_link_libraries(rambo_base PRIVATE rambo_base_lib)
    
    # Optimization flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(rambo_base PRIVATE $<$<CONFIG:Release>:-O3>)
    endif()
    
    if(UNIX AND NOT APPLE)
        target_link_libraries(rambo_base PRIVATE m)
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================
if(RAMBO_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp"
    )
    
    install(TARGETS rambo_base_lib
        EXPORT rambo-base-targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    
    install(EXPORT rambo-base-targets
        FILE rambo-base-targets.cmake
        NAMESPACE rambo::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-base
    )
    
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rambo-baseConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-baseConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-base
    )
    
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-baseConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-baseConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rambo-baseConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rambo-base
    )
endif()

message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
