# PhirstInstall.cmake — CMake install/export rules and configuration summary.
# Included at the end of CMakeLists.txt after the backend block has run.
# Expects _PHIRST_EXPORT_TARGET to be set by the backend include file.

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
        set(PHIRST_RESOLVED_GPU_ARCH "${PHIRST_SYCL_CUDA_ARCH}")
    elseif(PHIRST_SYCL_BACKEND STREQUAL "AMD")
        set(PHIRST_RESOLVED_GPU_ARCH "${PHIRST_SYCL_AMD_ARCH}")
    else()
        set(PHIRST_RESOLVED_GPU_ARCH "")  # JIT, no specific arch
    endif()
elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    if(CMAKE_CUDA_ARCHITECTURES)
        set(PHIRST_RESOLVED_GPU_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    elseif(CMAKE_HIP_ARCHITECTURES)
        set(PHIRST_RESOLVED_GPU_ARCH "${CMAKE_HIP_ARCHITECTURES}")
    elseif(PHIRST_ALPAKA_SYCL_DEVICE)
        set(PHIRST_RESOLVED_GPU_ARCH "${PHIRST_ALPAKA_SYCL_DEVICE}")
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
    elseif(PHIRST_ALPAKA_SYCL_DEVICE)
        set(_phirst_arch_summary "${PHIRST_ALPAKA_SYCL_DEVICE} (SYCL/oneAPI)")
    else()
        set(_phirst_arch_summary "CPU")
    endif()
elseif(PHIRST_BACKEND STREQUAL "SYCL")
    if(PHIRST_SYCL_BACKEND STREQUAL "CUDA")
        if(PHIRST_SYCL_CUDA_ARCH)
            set(_phirst_arch_summary "${PHIRST_SYCL_CUDA_ARCH}")
        else()
            set(_phirst_arch_summary "JIT (PTX)")
        endif()
    elseif(PHIRST_SYCL_BACKEND STREQUAL "AMD")
        set(_phirst_arch_summary "${PHIRST_SYCL_AMD_ARCH}")
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
