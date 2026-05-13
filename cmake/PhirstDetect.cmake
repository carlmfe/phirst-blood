# PhirstDetect.cmake — GPU hardware detection helpers (backend-native only).
# Included at the top of CMakeLists.txt. Provides:
#   detect_amd_architectures(out_var)
#   sycl_probe_intel_gpu()
#   normalize_nvidia_arch_numeric(arch_in out_var)
#   normalize_nvidia_arch_sycl(arch_in out_var)
#
# Detection philosophy: each function uses ONLY tools that ship with the
# backend's own installation, accessed via paths the backend itself advertises.
# No PATH searches for system-level tools (e.g. no nvidia-smi).
#
# NVIDIA (CUDA/Alpaka-CUDA): CMAKE_CUDA_ARCHITECTURES="native" is used — nvcc
#   queries the GPU driver at compile time.  Requires CMake >= 3.24.  For
#   cross-compilation set -DPHIRST_GPU_ARCH=<SM>.
#
# AMD (HIP, Alpaka-HIP, SYCL/AMD): detect_amd_architectures() queries
#   rocm_agent_enumerator via $ENV{ROCM_PATH} (set by the ROCm module) or
#   the standard /opt/rocm prefix.  rocminfo is used as a fallback.
#
# Intel SYCL: sycl_probe_intel_gpu() uses sycl-ls (available in both DPC++
#   and AdaptiveCpp) for informational output only.  Intel targets use JIT
#   by default; no GPU arch is required at configure time.

# ---------------------------------------------------------------------------
# AMD: query rocm_agent_enumerator or rocminfo via ROCm-native paths only.
# Uses $ENV{ROCM_PATH} (set by the ROCm module system) and /opt/rocm.
# No general PATH search — only paths advertised by the ROCm installation.
# Returns a semicolon-separated list of GFX targets, e.g. "gfx906;gfx90a",
# or sets out_var to "" if no AMD GPU is detected.
# ---------------------------------------------------------------------------
function(detect_amd_architectures out_var)
    # rocm_agent_enumerator — ships with ROCm's rocminfo package.
    # Search only the ROCm-native locations (env var set by module + standard prefix).
    set(_rocm_enum "")
    foreach(_candidate
            "$ENV{ROCM_PATH}/bin/rocm_agent_enumerator"
            "/opt/rocm/bin/rocm_agent_enumerator")
        if(EXISTS "${_candidate}")
            set(_rocm_enum "${_candidate}")
            break()
        endif()
    endforeach()

    if(_rocm_enum)
        execute_process(
            COMMAND "${_rocm_enum}" -t GPU
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
            if(_list)
                set(${out_var} "${_list}" PARENT_SCOPE)
                message(STATUS "Detected AMD GFX architectures: ${_list}")
                return()
            endif()
        endif()
    endif()

    # Fallback: rocminfo (also ships with ROCm; parses gfxNNN strings).
    set(_rocminfo "")
    foreach(_candidate
            "$ENV{ROCM_PATH}/bin/rocminfo"
            "/opt/rocm/bin/rocminfo")
        if(EXISTS "${_candidate}")
            set(_rocminfo "${_candidate}")
            break()
        endif()
    endforeach()

    if(_rocminfo)
        execute_process(
            COMMAND "${_rocminfo}"
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

    message(STATUS "ROCm tools not found in ROCM_PATH ($ENV{ROCM_PATH}) or /opt/rocm")
    set(${out_var} "" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Intel SYCL: informational probe via sycl-ls (provided by both DPC++ and
# AdaptiveCpp).  Does NOT set an architecture — Intel targets use JIT by
# default; no GPU arch is required at configure time.
# ---------------------------------------------------------------------------
function(sycl_probe_intel_gpu)
    find_program(_SYCL_LS sycl-ls)
    if(_SYCL_LS)
        execute_process(
            COMMAND "${_SYCL_LS}"
            OUTPUT_VARIABLE _raw
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _ret
            ERROR_QUIET
        )
        if(_ret EQUAL 0 AND _raw MATCHES "[Ii]ntel")
            message(STATUS "sycl-ls: Intel GPU device found (JIT will target it at runtime)")
        elseif(_ret EQUAL 0)
            message(STATUS "sycl-ls: no Intel GPU device listed")
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Arch normalization helpers
# ---------------------------------------------------------------------------

# Accept "89" or "sm_89"; return numeric (e.g. "89") for CUDA/Alpaka/HIP.
function(normalize_nvidia_arch_numeric arch_in out_var)
    string(REGEX REPLACE "^sm_" "" _n "${arch_in}")
    set(${out_var} "${_n}" PARENT_SCOPE)
endfunction()

# Accept "89" or "sm_89"; return prefixed (e.g. "sm_89") for SYCL/DPC++.
function(normalize_nvidia_arch_sycl arch_in out_var)
    string(REGEX REPLACE "^sm_" "" _n "${arch_in}")
    set(${out_var} "sm_${_n}" PARENT_SCOPE)
endfunction()
