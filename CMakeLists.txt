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
# Valid backends: AUTO, SERIAL, CUDA, KOKKOS, ALPAKA, SYCL
# Default: AUTO (probes for GPU toolchains in order: CUDA > HIP > SYCL > SERIAL)

set(PHIRST_BACKEND "AUTO" CACHE STRING "Backend: AUTO, SERIAL, CUDA, KOKKOS, ALPAKA, SYCL")
set_property(CACHE PHIRST_BACKEND PROPERTY STRINGS AUTO SERIAL CUDA KOKKOS ALPAKA SYCL)

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
            # ROCm present — prefer Kokkos if installed, else SYCL, else warn
            find_package(Kokkos QUIET)
            if(Kokkos_FOUND)
                set(PHIRST_BACKEND "KOKKOS")
                message(STATUS "AUTO: ROCm + Kokkos found → selecting KOKKOS backend")
            else()
                set(PHIRST_BACKEND "SYCL")
                set(SYCL_BACKEND "AMD" CACHE STRING "SYCL device backend (auto-set)" FORCE)
                message(STATUS "AUTO: ROCm found (no Kokkos) → selecting SYCL/AMD backend")
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

# Name of Drell-Yan example executable based on backend
string(TOLOWER "ex_drell_yan_${PHIRST_BACKEND}" EXENAME)

# =============================================================================
# Common Settings
# =============================================================================

# Default to Release build for good performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source files
set(SOURCES examples/drell_yan.cpp)

# =============================================================================
# Backend-Specific Configuration
# =============================================================================

if(PHIRST_BACKEND STREQUAL "SERIAL")
    # ---------------------------
    # Serial CPU Backend
    # ---------------------------
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    
    add_executable(${EXENAME} ${SOURCES})
    target_include_directories(${EXENAME} PRIVATE include)
    target_compile_definitions(${EXENAME} PRIVATE PHIRST_BACKEND_SERIAL)
    
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

    # Auto-detect SM architectures unless explicitly set by the user
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        detect_nvidia_architectures(_detected_cuda_archs)
        set(CMAKE_CUDA_ARCHITECTURES "${_detected_cuda_archs}" CACHE STRING "CUDA architectures" FORCE)
    endif()
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # For CUDA, rename drell_yan.cpp to drell_yan.cu
    set_source_files_properties(examples/drell_yan.cpp PROPERTIES LANGUAGE CUDA)
    
    add_executable(${EXENAME} ${SOURCES})
    target_include_directories(${EXENAME} PRIVATE include)
    target_compile_definitions(${EXENAME} PRIVATE PHIRST_BACKEND_CUDA)
    target_compile_options(${EXENAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
    
elseif(PHIRST_BACKEND STREQUAL "KOKKOS")
    # ---------------------------
    # Kokkos Backend
    # ---------------------------
    # Do NOT set CMAKE_CXX_STANDARD before find_package(Kokkos)!
    # Kokkos sets its own compiler flags based on how it was built.
    
    find_package(Kokkos REQUIRED)
    
    add_executable(${EXENAME} ${SOURCES})
    target_include_directories(${EXENAME} PRIVATE include)
    target_compile_definitions(${EXENAME} PRIVATE PHIRST_BACKEND_KOKKOS)
    target_link_libraries(${EXENAME} PRIVATE Kokkos::kokkos)
    
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
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
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
    
    alpaka_add_executable(${EXENAME} ${SOURCES})
    target_include_directories(${EXENAME} PRIVATE include)
    target_compile_definitions(${EXENAME} PRIVATE PHIRST_BACKEND_ALPAKA)
    target_link_libraries(${EXENAME} PRIVATE alpaka::alpaka)
    
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
    
    add_executable(${EXENAME} ${SOURCES})
    target_include_directories(${EXENAME} PRIVATE include)
    target_compile_definitions(${EXENAME} PRIVATE PHIRST_BACKEND_SYCL)
    
    if(SYCL_BACKEND STREQUAL "CUDA")
        set(CUDA_GPU_ARCH "sm_70" CACHE STRING "CUDA GPU architecture")
        target_compile_options(${EXENAME} PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
            -Xsycl-target-backend
            --cuda-gpu-arch=${CUDA_GPU_ARCH}
        )
        target_link_options(${EXENAME} PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
        )
    else()
        target_compile_options(${EXENAME} PRIVATE -fsycl)
        target_link_options(${EXENAME} PRIVATE -fsycl)
    endif()
    
else()
    message(FATAL_ERROR "Unknown backend: ${PHIRST_BACKEND}. Use SERIAL, CUDA, KOKKOS, ALPAKA, or SYCL.")
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
