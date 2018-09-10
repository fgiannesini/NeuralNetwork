package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentBatchNormProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnSoftMaxRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
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
                    {1, 1},
                    {1, 1}
            };
            double[] expectedGammaMatrix = {1, 1};

            double[] expectedBetaMatrix = {1.02, 1.04};

            LearningAlgorithm gradientDescent = new GradientDescent<>(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_one_hidden_layer_and_random_weights() {
            NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(2)
                    .addLayer(2, ActivationFunctionType.SOFT_MAX)
                    .buildBatchNormModel();

            DoubleMatrix input = DoubleMatrix.rand(2, 1);
            DoubleMatrix output = DoubleMatrix.rand(2, 1);

            LearningAlgorithm gradientDescent = new GradientDescent<>(neuralNetworkModel, new GradientDescentOnSoftMaxRegressionProcessProvider(new GradientDescentBatchNormProcessProvider()));
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.SOFT_MAX_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_one_hidden_layer() {
            NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .buildBatchNormModel();

            double[][] input = new double[][]{
                    {1, 2},
                    {3, 4}
            };
            double[][] output = new double[][]{
                    {3, 5},
                    {7, 9}
            };

            double[][] expectedWeightMatrix = {
                    {1, 1},
                    {1, 1}
            };
            double[] expectedGammaMatrix = {1.01, 1.01};

            double[] expectedBetaMatrix = {1.04, 1.06};

            LearningAlgorithm gradientDescent = new GradientDescent<>(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientWithDerivativeNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBetaMatrix)));
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
                    {2, 4},
                    {5, 6},
                    {9, 10}
            };

            double[][] output = new double[][]{
                    {11, 13},
                    {15, 17},
                    {19, 21}
            };

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

            LearningAlgorithm<BatchNormLayer> gradientDescent = new GradientDescent<>(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBetaMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBetaMatrix)));

            LearningAlgorithm<BatchNormLayer> gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}
