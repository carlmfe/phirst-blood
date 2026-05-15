cmake_minimum_required(VERSION 3.18)
project(PhirstDownstreamExample CXX)

# Demonstrates consuming phirst via find_package after cmake --install.
# Build with:
#   cmake --install <build-dir> --prefix /tmp/phirst-install
#   cmake -S examples/downstream -B /tmp/phirst-downstream-build \
#         -Dphirst_ROOT=/tmp/phirst-install
#   cmake --build /tmp/phirst-downstream-build

find_package(phirst REQUIRED)

# phirst::serial (or whichever backend was installed) is created by phirstConfig.cmake.
# The variable phirst_<backend>_FOUND tells you which one is available.
foreach(_b serial cuda kokkos alpaka sycl hip)
    if(phirst_${_b}_FOUND)
        set(_phirst_target "phirst::${_b}")
        break()
    endif()
endforeach()

if(NOT _phirst_target)
    message(FATAL_ERROR "No phirst backend target found after find_package(phirst)")
endif()

message(STATUS "Using phirst target: ${_phirst_target}")

add_executable(downstream_example main.cpp)
target_link_libraries(downstream_example PRIVATE ${_phirst_target})
