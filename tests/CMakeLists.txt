project(tests)

# '.lib' is the folder with Google Test sources
add_subdirectory(.lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(tests core_tests/random_test.cpp ../src/core/types.hpp ../src/core/random.hpp ../src/core/utility.hpp ../src/core/algebra.cpp ../src/core/algebra.hpp ../src/openn/io/nn_printer.cpp ../src/openn/io/nn_printer.hpp ../src/openn/io/nn_serializer.cpp ../src/openn/io/nn_serializer.hpp ../src/openn/openn.cpp ../src/openn/openn.hpp ../src/openn/training.cpp ../src/openn/training.hpp ../src/openn/types.hpp common/helpers.hpp common/ParamDB.cpp common/ParamDB.hpp core_tests/utility_test.cpp openn_tests/helpers.cpp openn_tests/helpers.hpp openn_tests/openn_test.cpp openn_tests/openn_test.hpp openn_tests/openn_test_instantiations.hpp openn_tests/openn_test_data.cpp openn_tests/openn_test_func.cpp)
target_link_libraries(tests gtest gtest_main)
