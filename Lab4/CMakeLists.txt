cmake_minimum_required(VERSION 3.7)
project(multip)

set(SOURCE_FILES matrix_multip.c)
add_executable(multip ${SOURCE_FILES})

find_package(GSL REQUIRED)
target_link_libraries(multip GSL::gsl GSL::gslcblas)