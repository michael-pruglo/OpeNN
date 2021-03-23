project(tests)

# '.lib' is the folder with Google Test sources
add_subdirectory(.lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(tests core_tests/random_test.cpp ../src/core/types.hpp ../src/core/random.hpp ../src/core/utility.hpp ../src/core/algebra.cpp ../src/core/algebra.hpp  ../src/openn/openn.cpp ../src/openn/openn.hpp ../src/openn/training.cpp ../src/openn/training.hpp ../src/openn/types.hpp common/helpers.hpp common/ParamDB_impl.hpp common/ParamDB.hpp core_tests/utility_test.cpp openn_tests/helpers.cpp openn_tests/helpers.hpp openn_tests/openn_test.cpp core_tests/algebra_test.cpp core_tests/types_test.cpp openn_tests/types_test.cpp)
target_link_libraries(tests gtest gtest_main)
