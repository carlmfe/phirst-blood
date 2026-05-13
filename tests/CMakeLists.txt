cmake_minimum_required(VERSION 3.18)
project(phirst_tests CXX)

enable_testing()

# Backend selection — inherits from parent cache when invoked via add_subdirectory();
# defaults to SERIAL for standalone usage (cmake -S tests -B build-tests).
set(PHIRST_BACKEND "SERIAL" CACHE STRING "Backend: SERIAL, CUDA, KOKKOS, ALPAKA, SYCL, HIP")

# Try system-installed GTest, otherwise use FetchContent to fetch googletest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

set(TEST_SOURCES
    test_phirst.cpp
    test_integrator.cpp
    test_integrands.cpp
    test_phase_space.cpp
    test_vegas.cpp
)

# Create test executable; Alpaka needs alpaka_add_executable for device compilation
if(PHIRST_BACKEND STREQUAL "ALPAKA")
    # alpaka_add_executable is available in this scope after find_package(alpaka) in parent
    alpaka_add_executable(test_phirst ${TEST_SOURCES})
else()
    # CUDA and HIP require sources to be compiled by their respective compilers
    if(PHIRST_BACKEND STREQUAL "CUDA")
        set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    elseif(PHIRST_BACKEND STREQUAL "HIP")
        set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()
    add_executable(test_phirst ${TEST_SOURCES})
endif()

# Use CMAKE_CURRENT_SOURCE_DIR so paths are correct both as standalone project
# (cmake -S tests) and as an add_subdirectory() call from the root.
target_include_directories(test_phirst PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Backend-specific compile definitions and link libraries
if(PHIRST_BACKEND STREQUAL "CUDA")
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_CUDA)
    target_compile_options(test_phirst PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
    )
elseif(PHIRST_BACKEND STREQUAL "KOKKOS")
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_KOKKOS)
    # Kokkos::kokkos is available because find_package(Kokkos) ran in the parent scope
    target_link_libraries(test_phirst PRIVATE Kokkos::kokkos)
elseif(PHIRST_BACKEND STREQUAL "SYCL")
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_SYCL)
    # SYCL_BACKEND, PHIRST_SYCL_CUDA_ARCH, PHIRST_SYCL_USE_ACPP and ACPP_EXTRA_COMPILE_OPTIONS
    # are all set by the parent scope via the universal SYCL_DRIVER detection.
    if(PHIRST_SYCL_USE_ACPP)
        # AdaptiveCpp: add_sycl_to_target() redirects compile/link via acpp.
        # ACPP_EXTRA_COMPILE_OPTIONS carries --acpp-targets=<backend>:<arch>.
        add_sycl_to_target(TARGET test_phirst)
    elseif(SYCL_BACKEND STREQUAL "CUDA")
        if(PHIRST_SYCL_CUDA_ARCH)
            target_compile_options(test_phirst PRIVATE
                -fsycl
                -fsycl-targets=nvptx64-nvidia-cuda
                -Xsycl-target-backend
                --cuda-gpu-arch=${PHIRST_SYCL_CUDA_ARCH}
            )
        else()
            target_compile_options(test_phirst PRIVATE
                -fsycl
                -fsycl-targets=nvptx64-nvidia-cuda
            )
        endif()
        target_link_options(test_phirst PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
        )
    elseif(SYCL_BACKEND STREQUAL "AMD")
        target_compile_options(test_phirst PRIVATE
            -fsycl
            -fsycl-targets=amdgcn-amd-amdhsa
            -Xsycl-target-backend
            --offload-arch=${PHIRST_SYCL_AMD_ARCH}
        )
        target_link_options(test_phirst PRIVATE
            -fsycl
            -fsycl-targets=amdgcn-amd-amdhsa
        )
    else()  # INTEL or any DPC++ JIT target
        target_compile_options(test_phirst PRIVATE -fsycl)
        target_link_options(test_phirst PRIVATE -fsycl)
    endif()
elseif(PHIRST_BACKEND STREQUAL "HIP")
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_HIP)
    # HIP language is already enabled in the parent scope via add_subdirectory()
elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_ALPAKA)
    target_link_libraries(test_phirst PRIVATE alpaka::alpaka)
else() # SERIAL (default) — also used for standalone cmake -S tests
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_SERIAL)
endif()

# Link against GTest::gtest (no bundled main) — test_phirst.cpp provides its own
# main() for Kokkos initialize/finalize. Linking gtest_main would cause a
# multiple-definition error.
if(TARGET GTest::gtest)
    target_link_libraries(test_phirst PRIVATE GTest::gtest)
else()
    target_link_libraries(test_phirst PRIVATE gtest)
endif()

include(GoogleTest)
gtest_discover_tests(test_phirst)
