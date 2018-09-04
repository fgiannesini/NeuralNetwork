package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentBatchNormProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class GradientDescentBatchNormTest {

    private IGradientDescentProcessProvider<BatchNormLayer> getGradientDescentProvider() {
        return new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentBatchNormProcessProvider());
    }

    @Nested
    class VariationOnInputAndLayerSize {
        @Test
        void learn_on_vector_with_one_hidden_layer() {
            NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .buildBatchNormModel();

            double[] input = new double[]{1, 2};
            double[] output = new double[]{3, 5};

            double[][] expectedWeightMatrix = {
                    {0.99, 1.01},
                    {0.98, 1.02}
            };
            double[] expectedGammaMatrix = {0.99, 1.01};

            double[] expectedBetaMatrix = {0.99, 1.01};

            LearningAlgorithm gradientDescent = new GradientDescent<>(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_two_hidden_layers() {
            NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .buildBatchNormModel();

            double[] input = new double[]{1, 2};
            double[] output = new double[]{10, 15};

            double[][] expectedFirstWeightMatrix = {
                    {0.99, 0.99, 0.99},
                    {0.98, 0.98, 0.98}
            };
            double[] expectedFirstGammaMatrix = {0.99, 0.99, 0.99};
            double[] expectedFirstBetaMatrix = {0.99, 0.99, 0.99};

            double[][] expectedSecondWeightMatrix = {
                    {0.88, 1.08},
                    {0.88, 1.08},
                    {0.88, 1.08}
            };
            double[] expectedSecondGammaMatrix = {0.97, 1.02};
            double[] expectedSecondBetaMatrix = {0.97, 1.02};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBetaMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers() {
            NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .buildBatchNormModel();

            double[][] input = new double[][]{
                    {1, 2},
                    {3, 4}
            };

            double[][] output = new double[][]{
                    {15, 15},
                    {20, 20}
            };

            double[][] expectedFirstWeightMatrix = {
                    {0.87, 0.87, 0.87},
                    {0.84, 0.84, 0.84}
            };
            double[] expectedFirstGammaMatrix = {0.97, 0.97, 0.97};
            double[] expectedFirstBetaMatrix = {0.97, 0.97, 0.97};

            double[][] expectedSecondWeightMatrix = {
                    {0.84, 0.84},
                    {0.84, 0.84},
                    {0.84, 0.84}
            };
            double[] expectedSecondGammaMatrix = {0.985, 0.985};
            double[] expectedSecondBetaMatrix = {0.985, 0.985};

            LearningAlgorithm gradientDescent = new GradientDescent<>(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBetaMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}
