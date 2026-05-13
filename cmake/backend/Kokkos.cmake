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
