# PhirstDetect.cmake — GPU hardware auto-detection and normalization helpers.
# Included at the top of CMakeLists.txt. Provides:
#   detect_nvidia_architectures(out_var)
#   detect_amd_architectures(out_var)
#   detect_intel_gpu_targets(out_var)
#   normalize_nvidia_arch_numeric(arch_in out_var)
#   normalize_nvidia_arch_sycl(arch_in out_var)
#   phirst_detect_hardware(backend out_arch)

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
