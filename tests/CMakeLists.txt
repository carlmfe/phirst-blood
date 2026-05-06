cmake_minimum_required(VERSION 3.18)
project(phirst_tests CXX)

enable_testing()

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

add_executable(test_phirst test_phirst.cpp test_integrator.cpp test_integrands.cpp)
# Use CMAKE_CURRENT_SOURCE_DIR so paths are correct both as standalone project
# (cmake -S tests) and as an add_subdirectory() call from the root.
target_include_directories(test_phirst PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
# Force serial backend for tests
target_compile_definitions(test_phirst PRIVATE PHIRST_BACKEND_SERIAL)

if(TARGET GTest::gtest_main)
    target_link_libraries(test_phirst PRIVATE GTest::gtest_main)
else()
    target_link_libraries(test_phirst PRIVATE gtest_main)
endif()

include(GoogleTest)
gtest_discover_tests(test_phirst)
