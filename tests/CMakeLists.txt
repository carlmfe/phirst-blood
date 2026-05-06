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

# CUDA and HIP require sources to be compiled by their respective compilers
if(PHIRST_BACKEND STREQUAL "CUDA")
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
elseif(PHIRST_BACKEND STREQUAL "HIP")
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

add_executable(test_phirst ${TEST_SOURCES})

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
    # SYCL_BACKEND and CUDA_GPU_ARCH are inherited from the parent scope
    if(SYCL_BACKEND STREQUAL "CUDA")
        target_compile_options(test_phirst PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
            -Xsycl-target-backend
            --cuda-gpu-arch=${CUDA_GPU_ARCH}
        )
        target_link_options(test_phirst PRIVATE
            -fsycl
            -fsycl-targets=nvptx64-nvidia-cuda
        )
    else()
        target_compile_options(test_phirst PRIVATE -fsycl)
        target_link_options(test_phirst PRIVATE -fsycl)
    endif()
elseif(PHIRST_BACKEND STREQUAL "HIP")
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_HIP)
    # HIP language is already enabled in the parent scope via add_subdirectory()
elseif(PHIRST_BACKEND STREQUAL "ALPAKA")
    # Alpaka test integration is deferred: alpaka_add_executable() is only available
    # after find_package(alpaka) in the parent scope and cannot be retrofitted onto a
    # plain add_executable(). Tests fall back to SERIAL for now.
    message(STATUS "Alpaka GPU test integration not yet supported — tests use SERIAL backend.")
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_SERIAL)
else() # SERIAL (default) — also used for standalone cmake -S tests
    target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_SERIAL)
endif()

if(TARGET GTest::gtest_main)
    target_link_libraries(test_phirst PRIVATE GTest::gtest_main)
else()
    target_link_libraries(test_phirst PRIVATE gtest_main)
endif()

include(GoogleTest)
gtest_discover_tests(test_phirst)
