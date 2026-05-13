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
