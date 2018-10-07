package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnSoftMaxRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

public class GradientDescentConvolutionTest {

    private IGradientDescentProcessProvider getGradientDescentProvider() {
        return new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider());
    }

    @Nested
    class VariationOnInputAndLayerSize {

        @Test
        void learn_with_one_fully_connected_layer() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(10, 10, 1)
                    .addFullyConnectedLayer(2, ActivationFunctionType.RELU)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new WeightBiasData(DoubleMatrix.rand(100, 1));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 1, 0));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_with_one_convolution_layer_and_one_fully_connected_layer() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(10, 10, 1)
                    .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.RELU)
                    .addFullyConnectedLayer(2, ActivationFunctionType.RELU)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.rand(10, 10)));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 1, 0));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_with_one_average_pooling_layer_and_one_fully_connected_layer() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(10, 10, 1)
                    .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.NONE)
                    .addAveragePoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                    .addFullyConnectedLayer(2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.rand(10, 10)));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 250, 200));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_with_one_max_pooling_layer_and_one_fully_connected_layer() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(10, 10, 1)
                    .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.NONE)
                    .addMaxPoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                    .addFullyConnectedLayer(2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.rand(10, 10)));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 250, 200));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_with_one_hidden_layer_and_one_of_each_type_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(10, 10, 1)
                    .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.NONE)
                    .addAveragePoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                    .addFullyConnectedLayer(2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(10, 10)));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 350, 370));

            double[][] expectedConvolutionWeightMatrix = {
                    {0.28, 0.28, 0.28},
                    {0.28, 0.28, 0.28},
                    {0.28, 0.28, 0.28}
            };

            double[] expectedConvolutionBiasMatrix = {0.28};

            double[][] expectedConnectedWeightMatrix = {
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0},
                    {-0.1, 0}
            };

            double[] expectedConnectedBiasMatrix = {0.89, 1.09};

//            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
//            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0,
                    Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedConvolutionWeightMatrix),
                            DataFormatConverter.fromTabToDoubleMatrix(expectedConvolutionBiasMatrix)));

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 2,
                    Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedConnectedWeightMatrix),
                            DataFormatConverter.fromTabToDoubleMatrix(expectedConnectedBiasMatrix)
                    )
            );

//            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
//            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
//            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_one_hidden_layer_with_softmax_layer_and_random_input() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(2)
                    .addBatchNormLayer(3, ActivationFunctionType.NONE)
                    .addBatchNormLayer(2, ActivationFunctionType.SOFT_MAX)
                    .buildNeuralNetworkModel();

            BatchNormData input = new BatchNormData(DoubleMatrix.rand(2, 3), new MeanDeviationProvider());
            DoubleMatrix rand = DoubleMatrix.rand(1, 3);
            BatchNormData output = new BatchNormData(DoubleMatrix.zeros(2, 3), null);
            output.getData().putRow(0, DoubleMatrix.ones(1, 3).subi(rand));
            output.getData().putRow(1, rand);

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnSoftMaxRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.SOFT_MAX_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);

            Assertions.fail("To be implemented");
        }

        @Test
        void learn_with_two_hidden_layers() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addBatchNormLayer(3, ActivationFunctionType.NONE)
                    .addBatchNormLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            MeanDeviationProvider meanDeviationProvider = new MeanDeviationProvider();
            LayerTypeData input = new BatchNormData(new DoubleMatrix(2, 3, 2, 4, 5, 6, 9, 10), meanDeviationProvider);
            LayerTypeData output = new BatchNormData(new DoubleMatrix(2, 3, 11, 13, 15, 17, 19, 21), meanDeviationProvider);

            double[][] expectedFirstWeightMatrix = {
                    {1.00008, 1.00008, 1.00008},
                    {0.99992, 0.999920, 0.99992}
            };
            double[] expectedFirstGammaMatrix = {1, 1, 1};
            double[] expectedFirstBetaMatrix = {1, 1, 1};

            double[][] expectedSecondWeightMatrix = {
                    {1, 1},
                    {1, 1},
                    {1, 1}
            };
            double[] expectedSecondGammaMatrix = {1.02237, 1.02237};
            double[] expectedSecondBetaMatrix = {1.14, 1.16};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBetaMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);

            Assertions.fail("To be implemented");
        }
    }

}
