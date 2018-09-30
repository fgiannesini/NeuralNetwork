package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.*;
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
        void learn_with_one_hidden_layer_and_one_of_each_type_channel() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(6, 6, 1)
                    .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.NONE)
                    .addAveragePoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                    .addFullyConnectedLayer(2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(10, 10)));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 3, 5));

            double[][] expectedConvolutionWeightMatrix = {
                    {1, 1, 1},
                    {1, 1, 1},
                    {1, 1, 1}
            };

            double[] expectedConvolutionBiasMatrix = {1};

            double[][] expectedConnectedWeightMatrix = {
                    {1, 1, 1, 1},
                    {1, 1, 1, 1}
            };

            double[] expectedConnectedBiasMatrix = {1, 1};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0,
                    Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedConvolutionWeightMatrix),
                            DataFormatConverter.fromTabToDoubleMatrix(expectedConvolutionBiasMatrix)));

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 3,
                    Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedConnectedWeightMatrix),
                            DataFormatConverter.fromTabToDoubleMatrix(expectedConnectedBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
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
            output.getInput().putRow(0, DoubleMatrix.ones(1, 3).subi(rand));
            output.getInput().putRow(1, rand);

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
