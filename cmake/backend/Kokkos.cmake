# Kokkos.cmake — Kokkos backend configuration (supports NVIDIA, AMD, CPU).
# Included by CMakeLists.txt when PHIRST_BACKEND=KOKKOS.
#
# Do NOT set CMAKE_CXX_STANDARD before find_package(Kokkos)!
# Kokkos sets its own compiler flags based on how it was built.

find_package(Kokkos REQUIRED)

# Report what this Kokkos installation targets (arch is baked into the install)
message(STATUS "Kokkos version: ${Kokkos_VERSION}")
message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")
if(Kokkos_ARCH)
    message(STATUS "Kokkos arch targets: ${Kokkos_ARCH}")
else()
    # Kokkos with GPU devices but no reported arch usually means the Kokkos install
    # was built with auto-arch (KOKKOS_ARCH_AUTO); the compiler will select at build time.
    foreach(_dev IN ITEMS "CUDA" "HIP" "SYCL")
        if(_dev IN_LIST Kokkos_DEVICES)
            message(WARNING
                "Kokkos_ARCH is empty but GPU device '${_dev}' is enabled.\n"
                "The Kokkos installation may have been built with KOKKOS_ARCH_AUTO=ON\n"
                "and will auto-detect the GPU arch at compile time.\n"
                "If you need a specific arch, rebuild Kokkos with "
                "-DKokkos_ARCH_<TARGET>=ON (e.g. -DKokkos_ARCH_AMPERE89=ON).")
            break()
        endif()
    endforeach()
endif()

add_library(phirst_kokkos INTERFACE)
add_library(phirst::kokkos ALIAS phirst_kokkos)
target_include_directories(phirst_kokkos INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(phirst_kokkos INTERFACE PHIRST_BACKEND_KOKKOS)
target_link_libraries(phirst_kokkos INTERFACE Kokkos::kokkos)
set(_PHIRST_EXPORT_TARGET phirst_kokkos)

if(PHIRST_BUILD_EXAMPLES)
    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_link_libraries(${EXENAME_DY} PRIVATE phirst::kokkos)

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_link_libraries(${EXENAME_EGG} PRIVATE phirst::kokkos)
endif()

if(PHIRST_BUILD_TESTS)
    add_subdirectory(tests)
endif()
