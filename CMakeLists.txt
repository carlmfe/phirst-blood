cmake_minimum_required(VERSION 3.18)
project("Phirst Blood" CXX)

# =============================================================================
# GPU Architecture Auto-Detection Utilities
# =============================================================================

# --- NVIDIA: query nvidia-smi for compute capabilities ---
# Returns a semicolon-separated list of SM architectures, e.g. "80;89"
function(detect_nvidia_architectures out_var)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _raw
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _ret
            ERROR_QUIET
        )
        if(_ret EQUAL 0 AND _raw)
            # "8.9\n8.0" → strip dots → deduplicate → "89;80"
            string(REPLACE "." "" _nodot "${_raw}")
            string(REPLACE "\n" ";" _list "${_nodot}")
            string(REPLACE " " "" _list "${_list}")
            list(REMOVE_DUPLICATES _list)
            list(REMOVE_ITEM _list "")
            set(${out_var} "${_list}" PARENT_SCOPE)
            message(STATUS "Detected NVIDIA SM architectures: ${_list}")
            return()
        endif()
    endif()
    message(STATUS "nvidia-smi not found or returned no GPU — defaulting to 'native'")
    set(${out_var} "native" PARENT_SCOPE)
endfunction()

# --- AMD: query rocm_agent_enumerator (preferred) or rocminfo ---
# Returns a semicolon-separated list of GFX targets, e.g. "gfx906;gfx90a"
function(detect_amd_architectures out_var)
    # Preferred: rocm_agent_enumerator (ships with ROCm, fast)
    find_program(ROCM_AGENT_ENUM rocm_agent_enumerator
        PATHS /opt/rocm/bin $ENV{ROCM_PATH}/bin)
    if(ROCM_AGENT_ENUM)
        execute_process(
            COMMAND ${ROCM_AGENT_ENUM} -t GPU
            OUTPUT_VARIABLE _raw
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _ret
            ERROR_QUIET
        )
        if(_ret EQUAL 0 AND _raw)
            string(REPLACE "\n" ";" _list "${_raw}")
            string(REPLACE " " "" _list "${_list}")
            list(REMOVE_DUPLICATES _list)
            list(REMOVE_ITEM _list "")
            set(${out_var} "${_list}" PARENT_SCOPE)
            message(STATUS "Detected AMD GFX architectures: ${_list}")
            return()
        endif()
    endif()
    # Fallback: parse rocminfo for 'gfxNNN' strings
    find_program(ROCMINFO rocminfo
        PATHS /opt/rocm/bin $ENV{ROCM_PATH}/bin)
    if(ROCMINFO)
        execute_process(
            COMMAND ${ROCMINFO}
            OUTPUT_VARIABLE _raw
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _ret
            ERROR_QUIET
        )
        if(_ret EQUAL 0)
            string(REGEX MATCHALL "gfx[0-9a-fA-F]+" _list "${_raw}")
            list(REMOVE_DUPLICATES _list)
            list(REMOVE_ITEM _list "")
            if(_list)
                set(${out_var} "${_list}" PARENT_SCOPE)
                message(STATUS "Detected AMD GFX architectures (rocminfo): ${_list}")
                return()
            endif()
        endif()
    endif()
    message(STATUS "ROCm tools not found — no AMD architecture detected")
    set(${out_var} "" PARENT_SCOPE)
endfunction()

# --- Intel: detect presence and capability via sycl-ls or ocloc ---
# Sets out_var to a list of SYCL AOT targets, e.g. "spir64_gen" or specific device IDs.
# sycl-ls is available in both oneAPI DPC++ and AdaptiveCpp.
function(detect_intel_gpu_targets out_var)
    find_program(SYCL_LS sycl-ls)
    if(SYCL_LS)
        execute_process(
            COMMAND ${SYCL_LS}
            OUTPUT_VARIABLE _raw
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _ret
            ERROR_QUIET
        )
        if(_ret EQUAL 0 AND _raw MATCHES "[Ii]ntel")
            # Generic SPIR-V works on all Intel GPUs; use spir64_gen for AOT.
            # For full JIT (portable), spir64 is the safe choice.
            set(${out_var} "spir64_gen" PARENT_SCOPE)
            message(STATUS "Detected Intel GPU via sycl-ls — SYCL target: spir64_gen")
            return()
        endif()
    endif()
    # Fallback: ocloc can enumerate Intel GPU device IDs for full AOT
    find_program(OCLOC ocloc
        PATHS /opt/intel/oneapi/compiler/latest/linux/bin $ENV{ONEAPI_ROOT}/compiler/latest/linux/bin)
    if(OCLOC)
        message(STATUS "Detected Intel ocloc — SYCL target: spir64_gen")
        set(${out_var} "spir64_gen" PARENT_SCOPE)
        return()
    endif()
    message(STATUS "No Intel GPU tools found")
    set(${out_var} "" PARENT_SCOPE)
endfunction()

# --- Arch normalization helpers ---
# Accept "89" or "sm_89"; return numeric (e.g. "89") for CUDA/Alpaka/HIP
function(normalize_nvidia_arch_numeric arch_in out_var)
    string(REGEX REPLACE "^sm_" "" _n "${arch_in}")
    set(${out_var} "${_n}" PARENT_SCOPE)
endfunction()

# Accept "89" or "sm_89"; return prefixed (e.g. "sm_89") for SYCL/DPC++
function(normalize_nvidia_arch_sycl arch_in out_var)
    string(REGEX REPLACE "^sm_" "" _n "${arch_in}")
    set(${out_var} "sm_${_n}" PARENT_SCOPE)
endfunction()

# Unified hardware-detection entry point.
# backend: CUDA, HIP, SYCL_CUDA, SYCL_AMD, SYCL_INTEL
# out_arch: returned in native format for the backend
function(phirst_detect_hardware backend out_arch)
    if(backend STREQUAL "CUDA")
        detect_nvidia_architectures(_a)
        set(${out_arch} "${_a}" PARENT_SCOPE)
    elseif(backend STREQUAL "HIP")
        detect_amd_architectures(_a)
        set(${out_arch} "${_a}" PARENT_SCOPE)
    elseif(backend STREQUAL "SYCL_CUDA")
        detect_nvidia_architectures(_a)
        if(NOT _a STREQUAL "native")
            set(_sycl_archs "")
            foreach(_sm IN LISTS _a)
                normalize_nvidia_arch_sycl("${_sm}" _sm_prefixed)
                list(APPEND _sycl_archs "${_sm_prefixed}")
            endforeach()
            set(${out_arch} "${_sycl_archs}" PARENT_SCOPE)
        else()
            set(${out_arch} "" PARENT_SCOPE)
        endif()
    elseif(backend STREQUAL "SYCL_AMD")
        detect_amd_architectures(_a)
        set(${out_arch} "${_a}" PARENT_SCOPE)
    elseif(backend STREQUAL "SYCL_INTEL")
        detect_intel_gpu_targets(_a)
        set(${out_arch} "${_a}" PARENT_SCOPE)
    else()
        set(${out_arch} "" PARENT_SCOPE)
    endif()
endfunction()

# =============================================================================
# Backend Selection
# =============================================================================
# Usage: cmake -DPHIRST_BACKEND=CUDA ..
# Valid backends: AUTO, SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, HIP
# Default: AUTO (probes for GPU toolchains in order: CUDA > KOKKOS/HIP > SYCL > SERIAL)

set(PHIRST_BACKEND "AUTO" CACHE STRING "Backend: AUTO, SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, HIP")
set_property(CACHE PHIRST_BACKEND PROPERTY STRINGS AUTO SERIAL CUDA KOKKOS ALPAKA SYCL HIP)

# GPU architecture override (optional).
# CUDA/Alpaka:  single SM number or semicolon-separated list, e.g. "89" or "80;89"
# SYCL/CUDA:    full arch string, e.g. "sm_89"
# HIP:          GFX string or semicolon-separated list, e.g. "gfx1100" or "gfx908;gfx90a"
# Leave empty to let each backend auto-detect (default behaviour).
set(PHIRST_GPU_ARCH "" CACHE STRING "Override GPU architecture (e.g. 89 for CUDA/Alpaka, sm_89 for SYCL, gfx1100 for HIP)")

# --- AUTO detection ---
if(PHIRST_BACKEND STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(PHIRST_BACKEND "CUDA")
        message(STATUS "AUTO: CUDA compiler found → selecting CUDA backend")
    else()
        find_program(_ROCM_AGENT rocm_agent_enumerator
            PATHS /opt/rocm/bin $ENV{ROCM_PATH}/bin)
        find_program(_ROCMINFO rocminfo
            PATHS /opt/rocm/bin $ENV{ROCM_PATH}/bin)
        if(_ROCM_AGENT OR _ROCMINFO)
            # ROCm present — prefer Kokkos, then native HIP, then SYCL/AMD
            find_package(Kokkos QUIET)
            if(Kokkos_FOUND)
                set(PHIRST_BACKEND "KOKKOS")
                message(STATUS "AUTO: ROCm + Kokkos found → selecting KOKKOS backend")
            else()
                check_language(HIP)
                if(CMAKE_HIP_COMPILER)
                    set(PHIRST_BACKEND "HIP")
                    message(STATUS "AUTO: ROCm + HIP compiler found → selecting HIP backend")
                else()
                    set(PHIRST_BACKEND "SYCL")
                    set(SYCL_BACKEND "AMD" CACHE STRING "SYCL device backend (auto-set)" FORCE)
                    message(STATUS "AUTO: ROCm found (no Kokkos, no HIP compiler) → selecting SYCL/AMD backend")
                endif()
            endif()
        else()
            find_program(_SYCL_LS sycl-ls)
            if(_SYCL_LS)
                execute_process(COMMAND ${_SYCL_LS}
                    OUTPUT_VARIABLE _sycl_out OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                if(_sycl_out MATCHES "[Ii]ntel")
                    set(PHIRST_BACKEND "SYCL")
                    set(SYCL_BACKEND "INTEL" CACHE STRING "SYCL device backend (auto-set)" FORCE)
                    message(STATUS "AUTO: Intel GPU found via sycl-ls → selecting SYCL/Intel backend")
                else()
                    set(PHIRST_BACKEND "SERIAL")
                    message(STATUS "AUTO: sycl-ls found but no Intel GPU — falling back to SERIAL")
                endif()
            else()
                set(PHIRST_BACKEND "SERIAL")
                message(STATUS "AUTO: No GPU toolchain found → selecting SERIAL backend")
            endif()
        endif()
    endif()
endif()

message(STATUS "PHIRST Backend: ${PHIRST_BACKEND}")

# Name of examples based on backend
string(TOLOWER "ex_drell_yan_${PHIRST_BACKEND}" EXENAME_DY)
string(TOLOWER "ex_eggholder_${PHIRST_BACKEND}" EXENAME_EGG)

# =============================================================================
# Common Settings
# =============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Default to Release build for good performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Test suite (disabled by default; enable with -DPHIRST_BUILD_TESTS=ON)
option(PHIRST_BUILD_TESTS "Build the GoogleTest suite alongside the examples" OFF)
if(PHIRST_BUILD_TESTS)
    enable_testing()
endif()

# Examples: ON when building as top-level project, OFF when used as subdirectory
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(_phirst_default_examples ON)
else()
    set(_phirst_default_examples OFF)
endif()
option(PHIRST_BUILD_EXAMPLES "Build example executables" ${_phirst_default_examples})

# Variables for package config template (populated per-backend below)
set(PHIRST_SYCL_BACKEND "")
set(PHIRST_SYCL_CUDA_ARCH "")
set(PHIRST_SYCL_AMD_ARCH "")

# Source files
set(SOURCES_DY examples/drell_yan.cpp)
set(SOURCES_EGG examples/eggholder.cpp)

# =============================================================================
# Backend-Specific Configuration
# =============================================================================

if(PHIRST_BACKEND STREQUAL "SERIAL")
    # ---------------------------
    # Serial CPU Backend
    # ---------------------------
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    add_library(phirst_serial INTERFACE)
    add_library(phirst::serial ALIAS phirst_serial)
    target_include_directories(phirst_serial INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_definitions(phirst_serial INTERFACE PHIRST_BACKEND_SERIAL)
    set(_PHIRST_EXPORT_TARGET phirst_serial)

    if(PHIRST_BUILD_EXAMPLES)
        add_executable(${EXENAME_DY} ${SOURCES_DY})
        target_link_libraries(${EXENAME_DY} PRIVATE phirst::serial)

        add_executable(${EXENAME_EGG} ${SOURCES_EGG})
        target_link_libraries(${EXENAME_EGG} PRIVATE phirst::serial)
    endif()

    if(PHIRST_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

elseif(PHIRST_BACKEND STREQUAL "CUDA")
    # ---------------------------
    # CUDA Backend
    # ---------------------------
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Resolve CUDA architectures BEFORE enable_language(CUDA) so our value takes priority
    # over CMake's toolkit default (enable_language sets CMAKE_CUDA_ARCHITECTURES otherwise).
    if(PHIRST_GPU_ARCH)
        normalize_nvidia_arch_numeric("${PHIRST_GPU_ARCH}" _cuda_arch)
        set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch [user]: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        detect_nvidia_architectures(_detected_cuda_archs)
        set(CMAKE_CUDA_ARCHITECTURES "${_detected_cuda_archs}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA arch [auto]: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found. Use -DPHIRST_BACKEND=SERIAL for CPU build.")
    endif()
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(phirst_cuda INTERFACE)
    add_library(phirst::cuda ALIAS phirst_cuda)
    target_include_directories(phirst_cuda INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_definitions(phirst_cuda INTERFACE PHIRST_BACKEND_CUDA)
    target_compile_options(phirst_cuda INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
    set(_PHIRST_EXPORT_TARGET phirst_cuda)

    if(PHIRST_BUILD_EXAMPLES)
        set_source_files_properties(examples/drell_yan.cpp examples/eggholder.cpp PROPERTIES LANGUAGE CUDA)

        add_executable(${EXENAME_DY} ${SOURCES_DY})
        target_link_libraries(${EXENAME_DY} PRIVATE phirst::cuda)

        add_executable(${EXENAME_EGG} ${SOURCES_EGG})
        target_link_libraries(${EXENAME_EGG} PRIVATE phirst::cuda)
    endif()

    if(PHIRST_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

elseif(PHIRST_BACKEND STREQUAL "KOKKOS")
    # ---------------------------
    # Kokkos Backend
    # ---------------------------
    # Do NOT set CMAKE_CXX_STANDARD before find_package(Kokkos)!
    # Kokkos sets its own compiler flags based on how it was built.

    find_package(Kokkos REQUIRED)

    # Report what this Kokkos installation targets (arch is baked into the install)
    message(STATUS "Kokkos version: ${Kokkos_VERSION}")
    message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")
    if(Kokkos_ARCH)
        message(STATUS "Kokkos arch targets: ${Kokkos_ARCH}")
    endif()

    add_library(phirst_kokkos INTERFACE)
    add_library(phirst::kokkos ALIAS phirst_kokkos)
    target_include_directories(phirst_kokkos INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_definitions(phirst_kokkos INTERFACE PHIRST_BACKEND_KOKKOS)
    target_link_libraries(phirst_kokkos INTERFACE Kokkos::kokkos)
    set(_PHIRST_EXPORT_TARGET phirst_kokkos)

    if(PHIRST_BUILD_EXAMPLES)
        add_executable(${EXENAME_DY} ${SOURCES_DY})
        target_link_libraries(${EXENAME_DY} PRIVATE phirst::kokkos)

        add_executable(${EXENAME_EGG} ${SOURCES_EGG})
        target_link_libraries(${EXENAME_EGG} PRIVATE phirst::kokkos)
    endif()

    if(PHIRST_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    # ---------------------------
    # Alpaka Backend
    # ---------------------------
    # Detect which hardware backend this Alpaka installation was built for by
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
            detect_nvidia_architectures(_cuda_arch)
            set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}" CACHE STRING "CUDA architectures" FORCE)
            message(STATUS "Alpaka CUDA arch [auto]: ${CMAKE_CUDA_ARCHITECTURES}")
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

elseif(PHIRST_BACKEND STREQUAL "SYCL")
    # ---------------------------
    # SYCL Backend
    # ---------------------------
    # Supports two SYCL runtime drivers, selected via SYCL_DRIVER:
    #
    #   DPC++  — Intel DPC++ or any clang++ with -fsycl support (default when SYCL_DRIVER
    #             is not set and AdaptiveCpp is not found).  Flags (-fsycl, -fsycl-targets,
    #             --offload-arch, etc.) are applied to the INTERFACE library target and
    #             propagate automatically to all consumers.
    #
    #   ACPP   — AdaptiveCpp (acpp / syclcc).  Uses CMake's add_sycl_to_target() on each
    #             consuming executable because RULE_LAUNCH_COMPILE cannot be set on an
    #             INTERFACE target.  Architecture is passed via ACPP_EXTRA_COMPILE_OPTIONS
    #             (--acpp-targets=<backend>:<arch>) or inherited from the ACPP_TARGETS
    #             environment variable set by the module system.
    #
    # SYCL_DRIVER auto-detection order (when SYCL_DRIVER=AUTO):
    #   1. If CMAKE_CXX_COMPILER basename is "acpp" or "syclcc" → ACPP (required)
    #   2. Else if SYCL_BACKEND=AMD and find_package(AdaptiveCpp) succeeds → ACPP
    #      (ACPP is the natural AMD driver; for CUDA/INTEL default to DPC++)
    #   3. Otherwise → DPC++

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    set(SYCL_BACKEND "CUDA" CACHE STRING "SYCL device backend: CUDA, INTEL, AMD")
    set_property(CACHE SYCL_BACKEND PROPERTY STRINGS CUDA INTEL AMD)

    set(SYCL_DRIVER "AUTO" CACHE STRING "SYCL runtime driver: AUTO, ACPP (AdaptiveCpp), DPC++ (Intel/clang)")
    set_property(CACHE SYCL_DRIVER PROPERTY STRINGS AUTO ACPP DPC++)

    # ---- Detect driver ----
    set(_sycl_use_acpp OFF)
    if(SYCL_DRIVER STREQUAL "ACPP")
        find_package(AdaptiveCpp REQUIRED)
        set(_sycl_use_acpp ON)
    elseif(SYCL_DRIVER STREQUAL "DPC++")
        # Explicitly use DPC++ / clang -fsycl; do not probe for AdaptiveCpp.
    else()  # AUTO
        get_filename_component(_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
        if(_cxx_name MATCHES "^(acpp|syclcc)$")
            find_package(AdaptiveCpp REQUIRED)
            set(_sycl_use_acpp ON)
        elseif(SYCL_BACKEND STREQUAL "AMD")
            # Only probe for ACPP on AMD; for CUDA/INTEL DPC++ is the natural choice.
            find_package(AdaptiveCpp QUIET)
            if(AdaptiveCpp_FOUND)
                set(_sycl_use_acpp ON)
            endif()
        endif()
    endif()

    if(_sycl_use_acpp)
        message(STATUS "SYCL driver: AdaptiveCpp ${AdaptiveCpp_VERSION}")
    else()
        message(STATUS "SYCL driver: Intel DPC++ / clang -fsycl")
    endif()
    set(PHIRST_SYCL_USE_ACPP ${_sycl_use_acpp})

    message(STATUS "SYCL device backend: ${SYCL_BACKEND}")

    add_library(phirst_sycl INTERFACE)
    add_library(phirst::sycl ALIAS phirst_sycl)
    target_include_directories(phirst_sycl INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_definitions(phirst_sycl INTERFACE PHIRST_BACKEND_SYCL)
    set(_PHIRST_EXPORT_TARGET phirst_sycl)

    # ---- Hardware target: resolve arch and apply flags ----
    if(SYCL_BACKEND STREQUAL "CUDA")
        # Resolve SYCL/CUDA arch: PHIRST_GPU_ARCH (any format) > cached CUDA_GPU_ARCH > auto-detect.
        # SYCL requires "sm_XX" prefix form; "native" keyword is not supported.
        if(PHIRST_GPU_ARCH)
            normalize_nvidia_arch_sycl("${PHIRST_GPU_ARCH}" _sycl_cuda_arch)
            message(STATUS "SYCL CUDA arch [user]: ${_sycl_cuda_arch}")
        elseif(DEFINED CACHE{CUDA_GPU_ARCH} AND NOT "$CACHE{CUDA_GPU_ARCH}" STREQUAL "")
            set(_sycl_cuda_arch "$CACHE{CUDA_GPU_ARCH}")
            message(STATUS "SYCL CUDA arch [cached]: ${_sycl_cuda_arch}")
        else()
            phirst_detect_hardware("SYCL_CUDA" _sycl_cuda_arch)
            if(NOT _sycl_cuda_arch)
                message(FATAL_ERROR
                    "SYCL/CUDA backend: no NVIDIA GPU detected and PHIRST_GPU_ARCH not set. "
                    "Specify the target architecture: -DPHIRST_GPU_ARCH=sm_89")
            endif()
            list(GET _sycl_cuda_arch 0 _sycl_cuda_arch)  # SYCL targets a single arch
            message(STATUS "SYCL CUDA arch [auto]: ${_sycl_cuda_arch}")
        endif()
        set(CUDA_GPU_ARCH "${_sycl_cuda_arch}" CACHE STRING "CUDA GPU architecture for SYCL (e.g. sm_89)" FORCE)
        set(PHIRST_SYCL_BACKEND "CUDA")
        set(PHIRST_SYCL_CUDA_ARCH "${CUDA_GPU_ARCH}")

        if(_sycl_use_acpp)
            # acpp CUDA target: --acpp-targets=cuda:sm_XY
            set(ACPP_EXTRA_COMPILE_OPTIONS "--acpp-targets=cuda:${CUDA_GPU_ARCH}"
                CACHE STRING "AdaptiveCpp extra compile options" FORCE)
        else()
            target_compile_options(phirst_sycl INTERFACE
                -fsycl
                -fsycl-targets=nvptx64-nvidia-cuda
                -Xsycl-target-backend
                --cuda-gpu-arch=${CUDA_GPU_ARCH}
            )
            target_link_options(phirst_sycl INTERFACE
                -fsycl
                -fsycl-targets=nvptx64-nvidia-cuda
            )
        endif()

    elseif(SYCL_BACKEND STREQUAL "AMD")
        # Resolve AMD arch: PHIRST_GPU_ARCH > PHIRST_AMD_GPU_ARCH cache > auto-detect
        if(PHIRST_GPU_ARCH)
            set(_sycl_amd_arch "${PHIRST_GPU_ARCH}")
        elseif(DEFINED CACHE{PHIRST_AMD_GPU_ARCH} AND NOT "$CACHE{PHIRST_AMD_GPU_ARCH}" STREQUAL "")
            set(_sycl_amd_arch "$CACHE{PHIRST_AMD_GPU_ARCH}")
        else()
            phirst_detect_hardware("SYCL_AMD" _sycl_amd_arch)
        endif()
        set(PHIRST_AMD_GPU_ARCH "${_sycl_amd_arch}" CACHE STRING "AMD GPU architecture for SYCL/AMD (e.g. gfx1100)" FORCE)
        set(PHIRST_SYCL_BACKEND "AMD")
        set(PHIRST_SYCL_AMD_ARCH "${PHIRST_AMD_GPU_ARCH}")

        if(_sycl_use_acpp)
            # acpp AMD target: --acpp-targets=hip:gfxXXXX
            if(PHIRST_AMD_GPU_ARCH)
                set(ACPP_EXTRA_COMPILE_OPTIONS "--acpp-targets=hip:${PHIRST_AMD_GPU_ARCH}"
                    CACHE STRING "AdaptiveCpp extra compile options" FORCE)
                message(STATUS "SYCL AMD arch: ${PHIRST_AMD_GPU_ARCH} (--acpp-targets=hip:${PHIRST_AMD_GPU_ARCH})")
            else()
                message(STATUS "SYCL AMD: using ACPP_TARGETS from environment ($ENV{ACPP_TARGETS})")
            endif()
        else()
            if(NOT PHIRST_AMD_GPU_ARCH)
                message(WARNING "SYCL/AMD (DPC++): PHIRST_GPU_ARCH not set; defaulting to gfx1100. "
                                "Pass -DPHIRST_GPU_ARCH=<arch> for your target GPU.")
                set(PHIRST_AMD_GPU_ARCH "gfx1100")
            endif()
            message(STATUS "SYCL AMD arch: ${PHIRST_AMD_GPU_ARCH} (-fsycl-targets=amdgcn-amd-amdhsa)")
            target_compile_options(phirst_sycl INTERFACE
                -fsycl
                -fsycl-targets=amdgcn-amd-amdhsa
                -Xsycl-target-backend
                --offload-arch=${PHIRST_AMD_GPU_ARCH}
            )
            target_link_options(phirst_sycl INTERFACE
                -fsycl
                -fsycl-targets=amdgcn-amd-amdhsa
            )
        endif()

    elseif(SYCL_BACKEND STREQUAL "INTEL")
        set(PHIRST_SYCL_BACKEND "INTEL")

        if(_sycl_use_acpp)
            # AdaptiveCpp + Intel: arch strings are device-specific (e.g. level_zero:gpu).
            # If PHIRST_GPU_ARCH is given, pass it; otherwise let ACPP_TARGETS from the
            # environment (or acpp's default "generic" JIT) take effect.
            if(PHIRST_GPU_ARCH)
                set(ACPP_EXTRA_COMPILE_OPTIONS "--acpp-targets=${PHIRST_GPU_ARCH}"
                    CACHE STRING "AdaptiveCpp extra compile options" FORCE)
                message(STATUS "SYCL Intel arch: ${PHIRST_GPU_ARCH} (--acpp-targets=${PHIRST_GPU_ARCH})")
            else()
                message(STATUS "SYCL Intel (ACPP): using ACPP_TARGETS from environment ($ENV{ACPP_TARGETS})")
            endif()
        else()
            # DPC++: JIT compilation — no -fsycl-targets; icpx selects the best backend.
            target_compile_options(phirst_sycl INTERFACE -fsycl)
            target_link_options(phirst_sycl INTERFACE -fsycl)
        endif()

    else()
        message(FATAL_ERROR "Unknown SYCL_BACKEND '${SYCL_BACKEND}'. Valid values: CUDA, INTEL, AMD")
    endif()

    if(PHIRST_BUILD_EXAMPLES)
        add_executable(${EXENAME_DY} ${SOURCES_DY})
        target_link_libraries(${EXENAME_DY} PRIVATE phirst::sycl)
        if(_sycl_use_acpp)
            add_sycl_to_target(TARGET ${EXENAME_DY})
        endif()

        add_executable(${EXENAME_EGG} ${SOURCES_EGG})
        target_link_libraries(${EXENAME_EGG} PRIVATE phirst::sycl)
        if(_sycl_use_acpp)
            add_sycl_to_target(TARGET ${EXENAME_EGG})
        endif()
    endif()

    if(PHIRST_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

elseif(PHIRST_BACKEND STREQUAL "HIP")
    # ---------------------------
    # HIP Backend (AMD GPUs)
    # ---------------------------
    # Requires ROCm with CMake native HIP support (CMake ≥ 3.21).
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

else()
    message(FATAL_ERROR "Unknown backend: ${PHIRST_BACKEND}. Use SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, or HIP.")
endif()

# =============================================================================
# Installation
# =============================================================================

# Install headers (phirst library headers + contrib dependencies)
install(DIRECTORY include/phirst
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)
install(DIRECTORY include/contrib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export the active backend INTERFACE target so downstream find_package() works
install(TARGETS ${_PHIRST_EXPORT_TARGET}
    EXPORT phirst-targets
)
install(EXPORT phirst-targets
    FILE phirst-targets.cmake
    NAMESPACE phirst::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phirst
)

# Package config for find_package(phirst REQUIRED) support
string(TOLOWER "${PHIRST_BACKEND}" _phirst_backend_lower)

# Record the resolved GPU arch for the installed config
if(PHIRST_BACKEND STREQUAL "CUDA")
    set(PHIRST_RESOLVED_GPU_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
elseif(PHIRST_BACKEND STREQUAL "HIP")
    set(PHIRST_RESOLVED_GPU_ARCH "${CMAKE_HIP_ARCHITECTURES}")
elseif(PHIRST_BACKEND STREQUAL "SYCL")
    if(PHIRST_SYCL_BACKEND STREQUAL "CUDA")
        set(PHIRST_RESOLVED_GPU_ARCH "${CUDA_GPU_ARCH}")
    elseif(PHIRST_SYCL_BACKEND STREQUAL "AMD")
        set(PHIRST_RESOLVED_GPU_ARCH "${PHIRST_AMD_GPU_ARCH}")
    else()
        set(PHIRST_RESOLVED_GPU_ARCH "")  # JIT, no specific arch
    endif()
elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    if(CMAKE_CUDA_ARCHITECTURES)
        set(PHIRST_RESOLVED_GPU_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    elseif(CMAKE_HIP_ARCHITECTURES)
        set(PHIRST_RESOLVED_GPU_ARCH "${CMAKE_HIP_ARCHITECTURES}")
    else()
        set(PHIRST_RESOLVED_GPU_ARCH "")
    endif()
else()
    set(PHIRST_RESOLVED_GPU_ARCH "")
endif()

configure_package_config_file(
    cmake/phirstConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/phirstConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phirst
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/phirstConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/phirstConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/phirstConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phirst
)

# =============================================================================
# Configuration Summary
# =============================================================================

# Resolve human-readable arch string for the summary
set(_phirst_arch_summary "")
if(PHIRST_BACKEND STREQUAL "CUDA")
    set(_phirst_arch_summary "${CMAKE_CUDA_ARCHITECTURES}")
elseif(PHIRST_BACKEND STREQUAL "KOKKOS")
    if(Kokkos_ARCH)
        set(_phirst_arch_summary "${Kokkos_ARCH} (Kokkos-managed)")
    else()
        set(_phirst_arch_summary "${Kokkos_DEVICES} (Kokkos-managed)")
    endif()
elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    if(CMAKE_CUDA_ARCHITECTURES)
        set(_phirst_arch_summary "${CMAKE_CUDA_ARCHITECTURES} (CUDA)")
    elseif(CMAKE_HIP_ARCHITECTURES)
        set(_phirst_arch_summary "${CMAKE_HIP_ARCHITECTURES} (HIP)")
    else()
        set(_phirst_arch_summary "CPU")
    endif()
elseif(PHIRST_BACKEND STREQUAL "SYCL")
    if(PHIRST_SYCL_BACKEND STREQUAL "CUDA")
        set(_phirst_arch_summary "${CUDA_GPU_ARCH}")
    elseif(PHIRST_SYCL_BACKEND STREQUAL "AMD")
        set(_phirst_arch_summary "${PHIRST_AMD_GPU_ARCH}")
    else()
        set(_phirst_arch_summary "JIT (INTEL)")
    endif()
elseif(PHIRST_BACKEND STREQUAL "HIP")
    set(_phirst_arch_summary "${CMAKE_HIP_ARCHITECTURES}")
else()
    set(_phirst_arch_summary "N/A (CPU serial)")
endif()

message(STATUS "")
message(STATUS "===========================================")
message(STATUS " Phirst Blood configuration summary")
message(STATUS "-------------------------------------------")
message(STATUS " Backend:     ${PHIRST_BACKEND}")
if(PHIRST_BACKEND STREQUAL "SYCL")
    message(STATUS " SYCL hw:     ${SYCL_BACKEND}")
    if(_sycl_use_acpp)
        message(STATUS " SYCL driver: AdaptiveCpp ${AdaptiveCpp_VERSION}")
    else()
        message(STATUS " SYCL driver: DPC++ / clang -fsycl")
    endif()
endif()
message(STATUS " GPU arch:    ${_phirst_arch_summary}")
message(STATUS " Build type:  ${CMAKE_BUILD_TYPE}")
message(STATUS " Compiler:    ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " Tests:       ${PHIRST_BUILD_TESTS}")
message(STATUS " Examples:    ${PHIRST_BUILD_EXAMPLES}")
message(STATUS " Install:     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===========================================")
message(STATUS "")
