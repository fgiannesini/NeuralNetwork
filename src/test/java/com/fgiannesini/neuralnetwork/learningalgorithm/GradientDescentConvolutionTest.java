package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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
        void learn_with_one_convolution_layer_and_one_fully_connected_layer_and_stride() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(10, 10, 1)
                    .addConvolutionLayer(3, 0, 2, 1, ActivationFunctionType.RELU)
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
        void learn_with_one_of_each_type_one_channel_one_input() {
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

            DoubleMatrix expectedConnectedWeightMatrix = DoubleMatrix.zeros(2, 36);
            expectedConnectedWeightMatrix.putRow(0, DoubleMatrix.ones(1, 36).muli(-0.1));
            expectedConnectedWeightMatrix.putRow(1, DoubleMatrix.ones(1, 36).muli(1.9));

            double[] expectedConnectedBiasMatrix = {0.89, 1.09};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0,
                    Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedConvolutionWeightMatrix),
                            DataFormatConverter.fromTabToDoubleMatrix(expectedConvolutionBiasMatrix)));

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 2,
                    Arrays.asList(
                            expectedConnectedWeightMatrix,
                            DataFormatConverter.fromTabToDoubleMatrix(expectedConnectedBiasMatrix)
                    )
            );

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_with_one_of_each_type_three_channels_two_inputs() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(10, 10, 3)
                    .addConvolutionLayer(3, 0, 1, 2, ActivationFunctionType.NONE)
                    .addMaxPoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                    .addFullyConnectedLayer(2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new ConvolutionData(Arrays.asList(
                    DoubleMatrix.ones(10, 10),
                    DoubleMatrix.ones(10, 10).muli(2),
                    DoubleMatrix.ones(10, 10).muli(3),
                    DoubleMatrix.ones(10, 10).muli(11),
                    DoubleMatrix.ones(10, 10).muli(12),
                    DoubleMatrix.ones(10, 10).muli(13)
            ));

            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 3950, 3970, 23375, 23450));

            List<DoubleMatrix> expectedConvolutionMatrices = Arrays.asList(
                    DoubleMatrix.ones(3, 3).muli(46.18),
                    DoubleMatrix.ones(3, 3).muli(49.96),
                    DoubleMatrix.ones(3, 3).muli(53.74),
                    DoubleMatrix.ones(3, 3).muli(46.18),
                    DoubleMatrix.ones(3, 3).muli(49.96),
                    DoubleMatrix.ones(3, 3).muli(53.74),
                    DoubleMatrix.ones(1, 1).muli(4.78),
                    DoubleMatrix.ones(1, 1).muli(4.78)
            );

            DoubleMatrix expectedConnectedWeightMatrix = DoubleMatrix.zeros(2, 72);
            expectedConnectedWeightMatrix.putRow(0, DoubleMatrix.ones(1, 72).muli(-44.275));
            expectedConnectedWeightMatrix.putRow(1, DoubleMatrix.ones(1, 72).muli(83.1));

            double[] expectedConnectedBiasMatrix = {0.81499, 1.29};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 2,
                    Arrays.asList(
                            expectedConnectedWeightMatrix,
                            DataFormatConverter.fromTabToDoubleMatrix(expectedConnectedBiasMatrix)
                    )
            );

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedConvolutionMatrices);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}
