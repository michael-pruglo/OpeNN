#include <tests/common/helpers.hpp>
#include <openn/openn.hpp>

namespace openn::types
{
    TEST(OpennTest, NeuralNetComputationSmallSigmoidMultiInput)
    {
        FeedForwardNetwork nn({
            { .wnb={ {{-0.0169, 0.704, -0.1163}}, {0.} } }
        });
        expect_double_vec_eq(nn({22.,0.,1.}), (core::Vec{0.38}), 0.01);
        expect_double_vec_eq(nn({38.,1.,1.}), (core::Vec{0.49}), 0.01);
        expect_double_vec_eq(nn({26.,1.,1.}), (core::Vec{0.54}), 0.01);
        expect_double_vec_eq(nn({35.,1.,1.}), (core::Vec{0.50}), 0.01);
        expect_double_vec_eq(nn({35.,0.,1.}), (core::Vec{0.33}), 0.01);
        expect_double_vec_eq(nn({14.,1.,1.}), (core::Vec{0.59}), 0.01);
        expect_double_vec_eq(nn({25.,0.,1.}), (core::Vec{0.37}), 0.01);
        expect_double_vec_eq(nn({54.,0.,1.}), (core::Vec{0.26}), 0.01);
    }

    void test_forwardpass_intermediate_results(
        const Vec& input,
        const std::vector<FeedForwardNetwork::LayerInitValuesData>& data,
        const std::vector<Vec>& intermediate_results,
        float_t tolerance = 0.01
    )
    {
        for (size_t depth = 0; depth < data.size(); ++depth)
        {
            FeedForwardNetwork tnn({data.begin(), data.begin() + depth + 1U});
            expect_double_vec_eq(tnn(input), intermediate_results[depth], tolerance);
        }
    }

    TEST(OpennTest, NeuralNetComputationSmallSigmoid)
    {
        test_forwardpass_intermediate_results(
            {-1., 1.},
            {
                { .wnb={ {{.23, .69}, {.01, .99}, {.14, .74}},  {.94, .49, .68} } },
                { .wnb={ {{.39, .97, .54}},                     {.96}           } },
                { .wnb={ {{.56}, {.89}},                        {.87, .77}      } }
            },
            {
                {.8, .81, .78},
                {.92},
                {.8, .83}
            }
        );
    }

    TEST(OpennTest, NeuralNetComputationSmallReLU)
    {
        test_forwardpass_intermediate_results(
            {1.4, 1.3},
            {
                { ActivationFType::ReLU, { {{.24, .93}, {.79, .88}},                                          {.7, .65}           } },
                { ActivationFType::ReLU, { {{.03, .07}, {.12, .8}, {.99, .79}, {.92, .21}},                   {.06, .7, .62, .53} } },
                { ActivationFType::ReLU, { {{.44, .28, .21, .53}, {.5, .02, .29, .68}, {.58, .22, .71, .94}}, {.48, .95, .85}     } },
            },
            {
                {2.25, 2.89},
                {.33, 3.28, 5.13, 3.2},
                {4.33, 4.85, 8.42}
            }
        );
    }

    TEST(OpennTest, NeuralNetComputationSmallTanh)
    {
        test_forwardpass_intermediate_results(
            {1.4, 1.3},
            {
                { ActivationFType::TANH, {{{.93, -1.41},              {-1.36, -1.04}, {.23,  .04}, {3.49, .6}, {-.62, 2.31}}, {1.58,  4.14, 1.15, -.63, 1.38}  } },
                { ActivationFType::TANH, {{{.78, .37, -.84, .26, .4}, {2.3,   2.66, -2.45, 2.51, 2.88}},                      {-1.32, -3.57}                  } },
                { ActivationFType::TANH, {{{.63, -.03},               {-.84,  .77},   {-.03, -.99}},                          {.5,    -.03, 1.02}                } }
            },
            {
                {.78, .7, .91, 1., 1.},
                {-0.51, 1.},
                {.148, .823, .052}
            }
        );
    }

    TEST(OpennTest, NeuralNetComputationMeduimMixed)
    {
        test_forwardpass_intermediate_results(
            {1.4, 1.3},
            {
                { ActivationFType::TANH,    {{{.68,  .59},                  {.44,  .14},                  {.09, .32}, {.14, .61}, {.52, .65}}, {.23,  .43,  .26, .88, .17}   } },
                { ActivationFType::SIGMOID, {{{.13,  .96,  .02,  .32, .25}, {.94,  .49,  .5,   .46, .85}, {.38, .07,  -.01, .31, .01}},        {.65,  .57,  .15}             } },
                { ActivationFType::ReLU,    {{{-.07, -.17, .45},            {-.13, .01,  .4}},                                                 {-.12, -.18}                } },
                { ActivationFType::TANH,    {{{.6,   -.11},                 {.99,  .33},                  {.31, .69}, {.54, -.04}},            {0.,   -.01, 0.,  0.}          } },
                { ActivationFType::SIGMOID, {{{.01,  -.7,  -.35, .45},      {.1,   -.42, -.21, .27},      {.01, -.37, -.19, .24}},        {-1.7, -.82, .23}           } },
            },
            {
                {.96, .84, .67, .95, .94},
                {.89, .97, .71},
                {0., 0.},
                {0., -.01, 0., 0.},
                {.16, .31, .56}
            }
        );
    }
}
