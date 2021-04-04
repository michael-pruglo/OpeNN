cmake_minimum_required(VERSION 3.19)
project(openn)

set(CMAKE_CXX_STANDARD 20)

include_directories(. src packages)
add_subdirectory(src)
add_subdirectory(tests)

find_package(xtensor)
find_package(xtensor-blas)
target_link_libraries(openn xtensor xtensor-blas)
target_link_libraries(tests xtensor xtensor-blas)