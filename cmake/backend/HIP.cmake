# HIP.cmake — native AMD HIP backend configuration.
# Included by CMakeLists.txt when PHIRST_BACKEND=HIP.
# Requires ROCm with CMake native HIP support (CMake >= 3.21).
# Use: cmake -DPHIRST_BACKEND=HIP -DPHIRST_GPU_ARCH=gfx1100 ..

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Resolve HIP architectures BEFORE enable_language(HIP) so our value takes priority.
# Format: gfx* strings, e.g. "gfx1100" or "gfx908;gfx90a"
if(PHIRST_GPU_ARCH)
    set(CMAKE_HIP_ARCHITECTURES "${PHIRST_GPU_ARCH}" CACHE STRING "HIP architectures" FORCE)
    message(STATUS "HIP arch [user]: ${CMAKE_HIP_ARCHITECTURES}")
else()
    detect_amd_architectures(_detected_hip_archs)
    if(_detected_hip_archs)
        set(CMAKE_HIP_ARCHITECTURES "${_detected_hip_archs}" CACHE STRING "HIP architectures" FORCE)
        message(STATUS "HIP arch [auto]: ${CMAKE_HIP_ARCHITECTURES}")
    else()
        message(FATAL_ERROR
            "No AMD GPU detected and PHIRST_GPU_ARCH not set. "
            "Specify the target architecture: -DPHIRST_GPU_ARCH=gfx1100")
    endif()
endif()

include(CheckLanguage)
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR "HIP compiler not found. Install ROCm and ensure hipcc is on PATH.")
endif()
enable_language(HIP)

set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

add_library(phirst_hip INTERFACE)
add_library(phirst::hip ALIAS phirst_hip)
target_include_directories(phirst_hip INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(phirst_hip INTERFACE PHIRST_BACKEND_HIP)
set(_PHIRST_EXPORT_TARGET phirst_hip)

if(PHIRST_BUILD_EXAMPLES)
    set_source_files_properties(examples/drell_yan.cpp examples/eggholder.cpp
        PROPERTIES LANGUAGE HIP)

    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_link_libraries(${EXENAME_DY} PRIVATE phirst::hip)

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_link_libraries(${EXENAME_EGG} PRIVATE phirst::hip)
endif()

if(PHIRST_BUILD_TESTS)
    # HIP language is enabled above; tests/CMakeLists.txt will compile
    # test sources as HIP via set_source_files_properties.
    add_subdirectory(tests)
endif()
