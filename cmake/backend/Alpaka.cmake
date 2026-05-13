# Alpaka.cmake — Alpaka 2.0.0 backend configuration (CUDA, HIP, or CPU sub-backend).
# Included by CMakeLists.txt when PHIRST_BACKEND=ALPAKA.
#
# Detects which hardware sub-backend this Alpaka installation was built for by
# inspecting alpakaTargets.cmake — the CMake export artifact that records
# the compile-definition interface baked into the installed library.
# We search all standard CMake path sources so any discovery mechanism works
# (alpaka_ROOT env/cache, CMAKE_PREFIX_PATH, Alpaka_DIR, etc.).

set(_alpaka_search_dirs "")
# Alpaka_DIR: set by our module directly to .../lib/cmake/alpaka
if(DEFINED ENV{Alpaka_DIR})
    list(APPEND _alpaka_search_dirs "$ENV{Alpaka_DIR}")
endif()
if(DEFINED Alpaka_DIR)
    list(APPEND _alpaka_search_dirs "${Alpaka_DIR}")
endif()
# alpaka_ROOT → lib/cmake/alpaka
foreach(_root "$ENV{alpaka_ROOT}" "${alpaka_ROOT}")
    if(_root)
        list(APPEND _alpaka_search_dirs "${_root}/lib/cmake/alpaka")
    endif()
endforeach()
# CMAKE_PREFIX_PATH (CMake list variable and environment colon-list)
foreach(_prefix ${CMAKE_PREFIX_PATH})
    list(APPEND _alpaka_search_dirs "${_prefix}/lib/cmake/alpaka")
endforeach()
if(DEFINED ENV{CMAKE_PREFIX_PATH})
    string(REPLACE ":" ";" _env_cpp "$ENV{CMAKE_PREFIX_PATH}")
    foreach(_prefix ${_env_cpp})
        list(APPEND _alpaka_search_dirs "${_prefix}/lib/cmake/alpaka")
    endforeach()
    unset(_env_cpp)
endif()

set(_alpaka_targets_file "")
foreach(_dir ${_alpaka_search_dirs})
    if(EXISTS "${_dir}/alpakaTargets.cmake")
        set(_alpaka_targets_file "${_dir}/alpakaTargets.cmake")
        break()
    endif()
endforeach()
unset(_alpaka_search_dirs)

set(_alpaka_hw "CPU")
if(_alpaka_targets_file)
    file(READ "${_alpaka_targets_file}" _alpaka_targets_content)
    if(_alpaka_targets_content MATCHES "ALPAKA_ACC_GPU_CUDA_ENABLED")
        set(_alpaka_hw "CUDA")
    elseif(_alpaka_targets_content MATCHES "ALPAKA_ACC_GPU_HIP_ENABLED")
        set(_alpaka_hw "HIP")
    elseif(_alpaka_targets_content MATCHES "ALPAKA_ACC_SYCL_ENABLED")
        set(_alpaka_hw "SYCL")
    endif()
    unset(_alpaka_targets_content)
else()
    message(WARNING
        "Could not locate alpakaTargets.cmake — defaulting to CPU-only Alpaka. "
        "Set alpaka_ROOT, Alpaka_DIR, or add the Alpaka prefix to CMAKE_PREFIX_PATH.")
endif()
unset(_alpaka_targets_file)

message(STATUS "Alpaka hardware backend [detected]: ${_alpaka_hw}")

if(_alpaka_hw STREQUAL "CUDA")
    # Resolve CUDA architectures BEFORE enable_language so our value takes priority.
    if(PHIRST_GPU_ARCH)
        normalize_nvidia_arch_numeric("${PHIRST_GPU_ARCH}" _cuda_arch)
        set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Alpaka CUDA arch [user]: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Use CMake's "native" keyword — nvcc queries the GPU at compile time.
        # Requires CMake >= 3.24.  For cross-compilation set -DPHIRST_GPU_ARCH=<SM>.
        if(CMAKE_VERSION VERSION_LESS "3.24")
            message(FATAL_ERROR
                "Auto-detection of CUDA architecture requires CMake >= 3.24 "
                "(CMAKE_CUDA_ARCHITECTURES=native). "
                "Either upgrade CMake or set -DPHIRST_GPU_ARCH=<SM>.")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Alpaka CUDA arch [auto]: native (nvcc will query GPU at compile time)")
        message(WARNING
            "Alpaka CUDA arch 'native' requires a GPU on the build host. "
            "For cross-compilation, set -DPHIRST_GPU_ARCH=<SM number, e.g. 89>.")
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "Alpaka installation has CUDA support but no CUDA compiler was found. "
            "Load a CUDA module or ensure nvcc is on PATH.")
    endif()
    enable_language(CUDA)
    set(alpaka_ACC_GPU_CUDA_ENABLE ON)

elseif(_alpaka_hw STREQUAL "HIP")
    # Resolve HIP architectures BEFORE enable_language so our value takes priority.
    if(PHIRST_GPU_ARCH)
        set(CMAKE_HIP_ARCHITECTURES "${PHIRST_GPU_ARCH}" CACHE STRING "HIP architectures" FORCE)
        message(STATUS "Alpaka HIP arch [user]: ${CMAKE_HIP_ARCHITECTURES}")
    else()
        detect_amd_architectures(_hip_arch)
        if(_hip_arch)
            set(CMAKE_HIP_ARCHITECTURES "${_hip_arch}" CACHE STRING "HIP architectures" FORCE)
            message(STATUS "Alpaka HIP arch [auto]: ${CMAKE_HIP_ARCHITECTURES}")
        else()
            message(WARNING
                "No AMD GPU detected and PHIRST_GPU_ARCH not set. "
                "Pass -DPHIRST_GPU_ARCH=gfx<target> to specify the target architecture.")
        endif()
    endif()

    include(CheckLanguage)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
        message(FATAL_ERROR
            "Alpaka installation has HIP support but no HIP compiler was found. "
            "Load a ROCm module or ensure hipcc is on PATH.")
    endif()
    enable_language(HIP)
    set(alpaka_ACC_GPU_HIP_ENABLE ON)

elseif(_alpaka_hw STREQUAL "SYCL")
    set(alpaka_ACC_SYCL_ENABLE ON)
    # Compiler flags are propagated by alpaka::alpaka; no extra language step needed.
endif()
unset(_alpaka_hw)

find_package(alpaka REQUIRED)
add_library(phirst_alpaka INTERFACE)
add_library(phirst::alpaka ALIAS phirst_alpaka)
target_include_directories(phirst_alpaka INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(phirst_alpaka INTERFACE PHIRST_BACKEND_ALPAKA)
# Use BUILD_INTERFACE so alpaka::alpaka (non-IMPORTED alias) is not recorded in the
# install export — the installed phirstConfig.cmake.in re-creates the link via
# find_dependency(alpaka) + set_property.
target_link_libraries(phirst_alpaka INTERFACE $<BUILD_INTERFACE:alpaka::alpaka>)
set(_PHIRST_EXPORT_TARGET phirst_alpaka)

if(PHIRST_BUILD_EXAMPLES)
    alpaka_add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_link_libraries(${EXENAME_DY} PRIVATE phirst::alpaka)

    alpaka_add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_link_libraries(${EXENAME_EGG} PRIVATE phirst::alpaka)
endif()

if(PHIRST_BUILD_TESTS)
    add_subdirectory(tests)
endif()
