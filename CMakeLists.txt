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
    
    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_include_directories(${EXENAME_DY} PRIVATE include)
    target_compile_definitions(${EXENAME_DY} PRIVATE PHIRST_BACKEND_SERIAL)

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_include_directories(${EXENAME_EGG} PRIVATE include)
    target_compile_definitions(${EXENAME_EGG} PRIVATE PHIRST_BACKEND_SERIAL)

    if(PHIRST_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

elseif(PHIRST_BACKEND STREQUAL "CUDA")
    # ---------------------------
    # CUDA Backend
    # ---------------------------
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found. Use -DPHIRST_BACKEND=SERIAL for CPU build.")
    endif()
    enable_language(CUDA)
    
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Resolve CUDA architectures: PHIRST_GPU_ARCH > existing cache > auto-detect
    if(PHIRST_GPU_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES "${PHIRST_GPU_ARCH}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA architectures (from PHIRST_GPU_ARCH): ${CMAKE_CUDA_ARCHITECTURES}")
    elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        detect_nvidia_architectures(_detected_cuda_archs)
        set(CMAKE_CUDA_ARCHITECTURES "${_detected_cuda_archs}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "CUDA architectures (auto-detected): ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "CUDA architectures (cached): ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # For CUDA, rename .cpp to .cu
    set_source_files_properties(examples/drell_yan.cpp examples/eggholder.cpp PROPERTIES LANGUAGE CUDA)

    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_include_directories(${EXENAME_DY} PRIVATE include)
    target_compile_definitions(${EXENAME_DY} PRIVATE PHIRST_BACKEND_CUDA)
    target_compile_options(${EXENAME_DY} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_include_directories(${EXENAME_EGG} PRIVATE include)
    target_compile_definitions(${EXENAME_EGG} PRIVATE PHIRST_BACKEND_CUDA)
    target_compile_options(${EXENAME_EGG} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )

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
    
    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_include_directories(${EXENAME_DY} PRIVATE include)
    target_compile_definitions(${EXENAME_DY} PRIVATE PHIRST_BACKEND_KOKKOS)
    target_link_libraries(${EXENAME_DY} PRIVATE Kokkos::kokkos)

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_include_directories(${EXENAME_EGG} PRIVATE include)
    target_compile_definitions(${EXENAME_EGG} PRIVATE PHIRST_BACKEND_KOKKOS)
    target_link_libraries(${EXENAME_EGG} PRIVATE Kokkos::kokkos)

    if(PHIRST_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    # ---------------------------
    # Alpaka Backend
    # ---------------------------
    set(ALPAKA_BACKEND "CPU" CACHE STRING "Alpaka sub-backend: CUDA, CPU, OMP")
    set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS CUDA CPU OMP)
    
    message(STATUS "Alpaka sub-backend: ${ALPAKA_BACKEND}")
    
    include(CheckLanguage)
    check_language(CUDA)
    
    if(ALPAKA_BACKEND STREQUAL "CUDA" AND CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if(PHIRST_GPU_ARCH)
            set(CMAKE_CUDA_ARCHITECTURES "${PHIRST_GPU_ARCH}" CACHE STRING "CUDA architectures" FORCE)
            message(STATUS "Alpaka CUDA architectures (from PHIRST_GPU_ARCH): ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
        endif()
        set(alpaka_ACC_GPU_CUDA_ENABLE ON)
        add_compile_definitions(ALPAKA_USE_CUDA)
    elseif(ALPAKA_BACKEND STREQUAL "OMP")
        find_package(OpenMP REQUIRED)
        set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE ON)
        add_compile_definitions(ALPAKA_USE_CPU_OMP)
    else()
        set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)
        add_compile_definitions(ALPAKA_USE_CPU_SERIAL)
    endif()

    find_package(alpaka REQUIRED)

    alpaka_add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_include_directories(${EXENAME_DY} PRIVATE include)
    target_compile_definitions(${EXENAME_DY} PRIVATE PHIRST_BACKEND_ALPAKA)
    target_link_libraries(${EXENAME_DY} PRIVATE alpaka::alpaka)

    alpaka_add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_include_directories(${EXENAME_EGG} PRIVATE include)
    target_compile_definitions(${EXENAME_EGG} PRIVATE PHIRST_BACKEND_ALPAKA)
    target_link_libraries(${EXENAME_EGG} PRIVATE alpaka::alpaka)

    if(PHIRST_BUILD_TESTS)
        add_subdirectory(tests)
    endif()

elseif(PHIRST_BACKEND STREQUAL "SYCL")
    # ---------------------------
    # SYCL Backend
    # ---------------------------
    # Requires SYCL-enabled compiler (e.g., Intel DPC++, AdaptiveCpp)
    # Use: cmake -DCMAKE_CXX_COMPILER=clang++ -DPHIRST_BACKEND=SYCL ..
    
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    set(SYCL_BACKEND "CUDA" CACHE STRING "SYCL device backend: CUDA, INTEL")
    set_property(CACHE SYCL_BACKEND PROPERTY STRINGS CUDA INTEL)
    
    message(STATUS "SYCL device backend: ${SYCL_BACKEND}")
    
    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_include_directories(${EXENAME_DY} PRIVATE include)
    target_compile_definitions(${EXENAME_DY} PRIVATE PHIRST_BACKEND_SYCL)

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_include_directories(${EXENAME_EGG} PRIVATE include)
    target_compile_definitions(${EXENAME_EGG} PRIVATE PHIRST_BACKEND_SYCL)
    
    if(SYCL_BACKEND STREQUAL "CUDA")
        set(CUDA_GPU_ARCH "sm_70" CACHE STRING "CUDA GPU architecture for SYCL")
        if(PHIRST_GPU_ARCH)
            set(CUDA_GPU_ARCH "${PHIRST_GPU_ARCH}" CACHE STRING "CUDA GPU architecture for SYCL" FORCE)
            message(STATUS "SYCL CUDA arch (from PHIRST_GPU_ARCH): ${CUDA_GPU_ARCH}")
        else()
            message(STATUS "SYCL CUDA arch: ${CUDA_GPU_ARCH}")
        endif()
        target_compile_options(${EXENAME_DY} PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
            -Xsycl-target-backend
            --cuda-gpu-arch=${CUDA_GPU_ARCH}
        )
        target_link_options(${EXENAME_DY} PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
        )
        target_compile_options(${EXENAME_EGG} PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
            -Xsycl-target-backend
            --cuda-gpu-arch=${CUDA_GPU_ARCH}
        )
        target_link_options(${EXENAME_EGG} PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
        )
    else()
        target_compile_options(${EXENAME_DY} PRIVATE -fsycl)
        target_link_options(${EXENAME_DY} PRIVATE -fsycl)
        target_compile_options(${EXENAME_EGG} PRIVATE -fsycl)
        target_link_options(${EXENAME_EGG} PRIVATE -fsycl)
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

    include(CheckLanguage)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
        message(FATAL_ERROR "HIP compiler not found. Install ROCm and ensure hipcc is on PATH.")
    endif()
    enable_language(HIP)

    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    # Resolve HIP architectures: PHIRST_GPU_ARCH > existing cache > auto-detect
    # Format: gfx* strings, e.g. "gfx1100" or "gfx908;gfx90a"
    if(PHIRST_GPU_ARCH)
        set(CMAKE_HIP_ARCHITECTURES "${PHIRST_GPU_ARCH}" CACHE STRING "HIP architectures" FORCE)
        message(STATUS "HIP architectures (from PHIRST_GPU_ARCH): ${CMAKE_HIP_ARCHITECTURES}")
    elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
        detect_amd_architectures(_detected_hip_archs)
        if(_detected_hip_archs)
            set(CMAKE_HIP_ARCHITECTURES "${_detected_hip_archs}" CACHE STRING "HIP architectures" FORCE)
            message(STATUS "HIP architectures (auto-detected): ${CMAKE_HIP_ARCHITECTURES}")
        else()
            message(FATAL_ERROR
                "No AMD GPU detected and PHIRST_GPU_ARCH not set. "
                "Specify the target architecture: -DPHIRST_GPU_ARCH=gfx1100")
        endif()
    else()
        message(STATUS "HIP architectures (cached): ${CMAKE_HIP_ARCHITECTURES}")
    endif()

    # Compile .cpp sources as HIP (no file rename needed with CMake native HIP)
    set_source_files_properties(examples/drell_yan.cpp examples/eggholder.cpp
        PROPERTIES LANGUAGE HIP)

    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_include_directories(${EXENAME_DY} PRIVATE include)
    target_compile_definitions(${EXENAME_DY} PRIVATE PHIRST_BACKEND_HIP)

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_include_directories(${EXENAME_EGG} PRIVATE include)
    target_compile_definitions(${EXENAME_EGG} PRIVATE PHIRST_BACKEND_HIP)

    if(PHIRST_BUILD_TESTS)
        # HIP language is enabled above; tests/CMakeLists.txt will compile
        # test sources as HIP via set_source_files_properties.
        add_subdirectory(tests)
    endif()

else()
    message(FATAL_ERROR "Unknown backend: ${PHIRST_BACKEND}. Use SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, or HIP.")
endif()

# =============================================================================
# Installation (Optional)
# =============================================================================

include(GNUInstallDirs)

# Install headers
install(DIRECTORY include/phirst
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
