# CUDA.cmake — direct NVIDIA CUDA backend configuration.
# Included by CMakeLists.txt when PHIRST_BACKEND=CUDA.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Resolve CUDA architectures BEFORE enable_language(CUDA) so our value takes priority
# over CMake's toolkit default (enable_language sets CMAKE_CUDA_ARCHITECTURES otherwise).
if(PHIRST_GPU_ARCH)
    normalize_nvidia_arch_numeric("${PHIRST_GPU_ARCH}" _cuda_arch)
    set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}" CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "CUDA arch [user]: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # Use CMake's "native" keyword — nvcc queries the GPU at compile time.
    # This is CUDA-native; no external tools are required.
    # Requires CMake >= 3.24.  For cross-compilation (no GPU on build host)
    # set -DPHIRST_GPU_ARCH=<SM> explicitly.
    if(CMAKE_VERSION VERSION_LESS "3.24")
        message(FATAL_ERROR
            "Auto-detection of CUDA architecture requires CMake >= 3.24 "
            "(CMAKE_CUDA_ARCHITECTURES=native). "
            "Either upgrade CMake or set -DPHIRST_GPU_ARCH=<SM>.")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "CUDA arch [auto]: native (nvcc will query GPU at compile time)")
    message(WARNING
        "CUDA arch 'native' requires a GPU on the build host. "
        "For cross-compilation, set -DPHIRST_GPU_ARCH=<SM number, e.g. 89>.")
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
