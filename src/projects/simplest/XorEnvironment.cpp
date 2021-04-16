#include<src/projects/simplest/XorEnvironment.hpp>

using namespace projects::simplest;

void XorEnvironment::run()
{
    trainer.set_network(std::make_shared<openn::FeedForwardNetwork>(
        std::vector<size_t>{ 2, 2, 1 }
    ));
    trainer.set_hyper_parameters({
        .epochs     = 160,
        .eta        = 0.1,
        .method     = openn::TrainingMethod::FULL_GRAD_DESCENT,
    });
    const std::vector<openn::TrainingSample> trn_vec {
        openn::TrainingSample{ {0., 0.}, {0.} },
        openn::TrainingSample{ {0., 1.}, {1.} },
        openn::TrainingSample{ {1., 0.}, {1.} },
        openn::TrainingSample{ {1., 1.}, {0.} },
    };
    const std::vector<openn::TrainingSample> tst_vec {
        openn::TrainingSample{ {1., 0.}, {1.} },
        openn::TrainingSample{ {1., 1.}, {0.} },
    };
    trainer.set_training_data(&trn_vec);
    trainer.set_test_data(&tst_vec);

    trainer.train(true);
}
