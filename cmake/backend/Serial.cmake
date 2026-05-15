# Serial.cmake — SERIAL (CPU, single-thread) backend configuration.
# Included by CMakeLists.txt when PHIRST_BACKEND=SERIAL.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(phirst_serial INTERFACE)
add_library(phirst::serial ALIAS phirst_serial)
target_include_directories(phirst_serial INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(phirst_serial INTERFACE PHIRST_BACKEND_SERIAL)
set(_PHIRST_EXPORT_TARGET phirst_serial)

if(PHIRST_BUILD_EXAMPLES)
    add_executable(${EXENAME_DY} ${SOURCES_DY})
    target_link_libraries(${EXENAME_DY} PRIVATE phirst::serial)

    add_executable(${EXENAME_EGG} ${SOURCES_EGG})
    target_link_libraries(${EXENAME_EGG} PRIVATE phirst::serial)
endif()

if(PHIRST_BUILD_TESTS)
    add_subdirectory(tests)
endif()
