cmake_minimum_required(VERSION 3.10)
project(rambo)

set(CMAKE_CXX_STANDARD 11)

add_executable(rambo main.cpp rambo.cc)

target_include_directories(rambo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
