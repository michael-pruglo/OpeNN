project(tests)

# '.lib' is the folder with Google Test sources
add_subdirectory(.lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(tests ../src/core/types.hpp ../src/core/random.hpp ../src/core/utility.hpp ../src/core/algebra.cpp ../src/core/algebra.hpp  ../src/openn/openn.cpp ../src/openn/openn.hpp ../src/openn/training.cpp ../src/openn/training.hpp ../src/openn/types.hpp common/helpers.hpp common/ParamDB_impl.hpp common/ParamDB.hpp core_tests/utility_test.cpp openn_tests/helpers.cpp openn_tests/helpers.hpp openn_tests/openn_test.cpp core_tests/algebra_test.cpp openn_tests/types_test.cpp openn_tests/training_test.cpp ../src/openn/types.cpp ../src/misc/resource_unpacker.cpp ../src/misc/resource_unpacker.hpp misc_tests/resource_unpacker_test.cpp ../src/projects/mnist/types.hpp ../src/projects/mnist/TrainingEnvironment.cpp ../src/projects/mnist/TrainingEnvironment.hpp ../src/projects/simplest/XorEnvironment.cpp ../src/projects/simplest/XorEnvironment.hpp)
target_link_libraries(tests gtest gtest_main)
