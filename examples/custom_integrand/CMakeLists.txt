cmake_minimum_required(VERSION 3.18)
project(CustomIntegrand CXX)

set(PHIRST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "")
list(APPEND CMAKE_MODULE_PATH "${PHIRST_SOURCE_DIR}/cmake")
include_directories("${PHIRST_SOURCE_DIR}/include")
include(PhirstIntegrand)

phirst_add_integrand_module(
    NAME            my_integrand
    SOURCE          my_integrand.cpp
    INTEGRAND_TYPE  MyIntegrand
    PARTICLES       2
    BACKEND         SERIAL
)
