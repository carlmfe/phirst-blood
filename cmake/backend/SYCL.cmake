# SYCL.cmake — SYCL backend configuration (DPC++ or AdaptiveCpp driver).
# Included by CMakeLists.txt when PHIRST_BACKEND=SYCL.
#
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
# Detection philosophy: use only tools native to the selected driver.
# No PATH searches for system-level tools.
#
#   CUDA/DPC++:  AOT when PHIRST_GPU_ARCH set; JIT (PTX) otherwise + WARNING.
#                "native" is not supported by -fsycl — CUDA arch must be sm_XX.
#   CUDA/ACPP:   ACPP_TARGETS env (set by module) > PHIRST_GPU_ARCH > FATAL.
#   AMD/DPC++:   PHIRST_GPU_ARCH > detect_amd_architectures() > FATAL (always AOT).
#   AMD/ACPP:    ACPP_TARGETS env > PHIRST_GPU_ARCH > detect_amd_architectures() > FATAL.
#   INTEL/DPC++: JIT via Level Zero — no arch needed, always + WARNING.
#   INTEL/ACPP:  ACPP_TARGETS env > PHIRST_GPU_ARCH > JIT + WARNING.

if(SYCL_BACKEND STREQUAL "CUDA")
    set(PHIRST_SYCL_BACKEND "CUDA")

    if(_sycl_use_acpp)
        # ACPP CUDA: requires an explicit target (no PTX JIT in standard ACPP mode).
        if(DEFINED ENV{ACPP_TARGETS} AND NOT "$ENV{ACPP_TARGETS}" STREQUAL "")
            message(STATUS "SYCL CUDA arch [ACPP]: using ACPP_TARGETS from environment ($ENV{ACPP_TARGETS})")
            set(ACPP_EXTRA_COMPILE_OPTIONS ""
                CACHE STRING "AdaptiveCpp extra compile options" FORCE)
            # Extract arch from ACPP_TARGETS for the install config.
            string(REGEX MATCH "cuda:([^ ;,]+)" _acpp_cuda_match "$ENV{ACPP_TARGETS}")
            if(CMAKE_MATCH_1)
                normalize_nvidia_arch_sycl("${CMAKE_MATCH_1}" _sycl_cuda_arch)
            else()
                set(_sycl_cuda_arch "")
            endif()
        elseif(PHIRST_GPU_ARCH)
            normalize_nvidia_arch_sycl("${PHIRST_GPU_ARCH}" _sycl_cuda_arch)
            set(ACPP_EXTRA_COMPILE_OPTIONS "--acpp-targets=cuda:${_sycl_cuda_arch}"
                CACHE STRING "AdaptiveCpp extra compile options" FORCE)
            message(STATUS "SYCL CUDA arch [ACPP, user]: ${_sycl_cuda_arch}")
        else()
            message(FATAL_ERROR
                "SYCL/CUDA with AdaptiveCpp requires an explicit target architecture.\n"
                "Set -DPHIRST_GPU_ARCH=<SM, e.g. sm_89 or 89>, or set the\n"
                "ACPP_TARGETS environment variable (e.g. 'cuda:sm_89'),\n"
                "or load a module that sets ACPP_TARGETS automatically.")
        endif()
    else()
        # DPC++: AOT if PHIRST_GPU_ARCH given; PTX JIT otherwise.
        if(PHIRST_GPU_ARCH)
            normalize_nvidia_arch_sycl("${PHIRST_GPU_ARCH}" _sycl_cuda_arch)
            message(STATUS "SYCL CUDA arch [DPC++, user]: ${_sycl_cuda_arch}")
            target_compile_options(phirst_sycl INTERFACE
                -fsycl
                -fsycl-targets=nvptx64-nvidia-cuda
                -Xsycl-target-backend
                --cuda-gpu-arch=${_sycl_cuda_arch}
            )
        else()
            set(_sycl_cuda_arch "")
            message(STATUS "SYCL CUDA arch [DPC++]: JIT (PTX) — no PHIRST_GPU_ARCH set")
            message(WARNING
                "SYCL/CUDA (DPC++) will use PTX JIT compilation.\n"
                "The PTX binary is JIT-compiled by the CUDA driver at runtime.\n"
                "This is valid but a GPU must be present at runtime.\n"
                "For AOT compilation set -DPHIRST_GPU_ARCH=<SM, e.g. sm_89>.")
            target_compile_options(phirst_sycl INTERFACE
                -fsycl
                -fsycl-targets=nvptx64-nvidia-cuda
            )
        endif()
        target_link_options(phirst_sycl INTERFACE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
        )
    endif()
    set(PHIRST_SYCL_CUDA_ARCH "${_sycl_cuda_arch}")

elseif(SYCL_BACKEND STREQUAL "AMD")
    set(PHIRST_SYCL_BACKEND "AMD")
    # AMD targets always require AOT — no JIT path for amdgcn-amd-amdhsa.

    if(_sycl_use_acpp)
        # ACPP AMD: ACPP_TARGETS env > PHIRST_GPU_ARCH > detect_amd_architectures() > FATAL.
        if(DEFINED ENV{ACPP_TARGETS} AND NOT "$ENV{ACPP_TARGETS}" STREQUAL "")
            message(STATUS "SYCL AMD arch [ACPP]: using ACPP_TARGETS from environment ($ENV{ACPP_TARGETS})")
            set(ACPP_EXTRA_COMPILE_OPTIONS ""
                CACHE STRING "AdaptiveCpp extra compile options" FORCE)
            string(REGEX MATCH "hip:([^ ;,]+)" _acpp_hip_match "$ENV{ACPP_TARGETS}")
            set(_sycl_amd_arch "${CMAKE_MATCH_1}")
        elseif(PHIRST_GPU_ARCH)
            set(_sycl_amd_arch "${PHIRST_GPU_ARCH}")
            set(ACPP_EXTRA_COMPILE_OPTIONS "--acpp-targets=hip:${_sycl_amd_arch}"
                CACHE STRING "AdaptiveCpp extra compile options" FORCE)
            message(STATUS "SYCL AMD arch [ACPP, user]: ${_sycl_amd_arch}")
        else()
            detect_amd_architectures(_detected_amd)
            if(_detected_amd)
                list(GET _detected_amd 0 _sycl_amd_arch)
                set(ACPP_EXTRA_COMPILE_OPTIONS "--acpp-targets=hip:${_sycl_amd_arch}"
                    CACHE STRING "AdaptiveCpp extra compile options" FORCE)
                message(STATUS "SYCL AMD arch [ACPP, auto]: ${_sycl_amd_arch}")
            else()
                message(FATAL_ERROR
                    "SYCL/AMD with AdaptiveCpp requires an explicit target architecture.\n"
                    "Set -DPHIRST_GPU_ARCH=<gfxXXXX>, or set the ACPP_TARGETS\n"
                    "environment variable (e.g. 'hip:gfx1100'), or load a ROCm\n"
                    "module that sets ROCM_PATH.")
            endif()
        endif()
    else()
        # DPC++: PHIRST_GPU_ARCH > detect_amd_architectures() > FATAL.
        if(PHIRST_GPU_ARCH)
            set(_sycl_amd_arch "${PHIRST_GPU_ARCH}")
            message(STATUS "SYCL AMD arch [DPC++, user]: ${_sycl_amd_arch}")
        else()
            detect_amd_architectures(_detected_amd)
            if(_detected_amd)
                list(GET _detected_amd 0 _sycl_amd_arch)
                message(STATUS "SYCL AMD arch [DPC++, auto]: ${_sycl_amd_arch}")
            else()
                message(FATAL_ERROR
                    "SYCL/AMD (DPC++) requires an explicit target architecture.\n"
                    "Set -DPHIRST_GPU_ARCH=<gfxXXXX>, or load a ROCm module\n"
                    "that sets the ROCM_PATH environment variable.")
            endif()
        endif()
        target_compile_options(phirst_sycl INTERFACE
            -fsycl
            -fsycl-targets=amdgcn-amd-amdhsa
            -Xsycl-target-backend
            --offload-arch=${_sycl_amd_arch}
        )
        target_link_options(phirst_sycl INTERFACE
            -fsycl
            -fsycl-targets=amdgcn-amd-amdhsa
        )
    endif()
    set(PHIRST_SYCL_AMD_ARCH "${_sycl_amd_arch}")

elseif(SYCL_BACKEND STREQUAL "INTEL")
    set(PHIRST_SYCL_BACKEND "INTEL")
    # Intel SYCL uses Level Zero / OpenCL JIT by default for both drivers.

    if(_sycl_use_acpp)
        # ACPP Intel: ACPP_TARGETS env > PHIRST_GPU_ARCH > JIT + WARNING.
        if(DEFINED ENV{ACPP_TARGETS} AND NOT "$ENV{ACPP_TARGETS}" STREQUAL "")
            message(STATUS "SYCL Intel arch [ACPP]: using ACPP_TARGETS from environment ($ENV{ACPP_TARGETS})")
            set(ACPP_EXTRA_COMPILE_OPTIONS ""
                CACHE STRING "AdaptiveCpp extra compile options" FORCE)
        elseif(PHIRST_GPU_ARCH)
            set(ACPP_EXTRA_COMPILE_OPTIONS "--acpp-targets=${PHIRST_GPU_ARCH}"
                CACHE STRING "AdaptiveCpp extra compile options" FORCE)
            message(STATUS "SYCL Intel arch [ACPP, user]: ${PHIRST_GPU_ARCH}")
        else()
            # JIT via SSCP / Level Zero — ACPP default for Intel.
            message(STATUS "SYCL Intel [ACPP]: JIT via Level Zero (no explicit target set)")
            message(WARNING
                "SYCL/Intel with AdaptiveCpp will use generic JIT compilation.\n"
                "To target a specific device set -DPHIRST_GPU_ARCH=<acpp-target-string>\n"
                "or set the ACPP_TARGETS environment variable.")
        endif()
    else()
        # DPC++: JIT is the natural default for Intel GPUs.
        target_compile_options(phirst_sycl INTERFACE -fsycl)
        target_link_options(phirst_sycl INTERFACE -fsycl)
        message(STATUS "SYCL Intel [DPC++]: JIT via Level Zero")
        message(WARNING
            "SYCL/Intel (DPC++) will use JIT compilation.\n"
            "The kernel is compiled by the Intel GPU driver at runtime.\n"
            "This is the standard Intel SYCL workflow.")
    endif()
    sycl_probe_intel_gpu()  # informational only

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
