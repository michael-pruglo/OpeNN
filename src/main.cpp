#include <projects/mnist/TrainingEnvironment.hpp>

int main()
{
    projects::mnist::TrainingEnvironment env;

    const std::string PATH_PREF = "../../resources/mnist/";
    env.init_training_data(PATH_PREF + "train-images.idx3-ubyte", PATH_PREF + "train-labels.idx1-ubyte");
    env.init_test_data    (PATH_PREF + "t10k-images.idx3-ubyte",  PATH_PREF + "t10k-labels.idx1-ubyte");

    env.display_training_sample(4517);
    env.display_validating_sample(1400);
    env.display_testing_sample(2100);
}
