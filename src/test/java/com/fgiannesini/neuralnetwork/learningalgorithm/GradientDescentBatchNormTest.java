package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentBatchNormProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
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
            double[] expectedBiasMatrix = {0.99, 1.01};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBiasMatrix)));

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
            double[] expectedFirstBiasMatrix = {0.99, 0.99, 0.99};

            double[][] expectedSecondWeightMatrix = {
                    {0.88, 1.08},
                    {0.88, 1.08},
                    {0.88, 1.08}
            };
            double[] expectedSecondBiasMatrix = {0.97, 1.02};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

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
            double[] expectedFirstBiasMatrix = {0.97, 0.97, 0.97};

            double[][] expectedSecondWeightMatrix = {
                    {0.84, 0.84},
                    {0.84, 0.84},
                    {0.84, 0.84}
            };
            double[] expectedSecondBiasMatrix = {0.985, 0.985};
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

    @Nested
    class DataExpectation {

        @Test
        void learn_on_vector_with_two_hidden_layers_and_sigmoid_activation_function() {
            NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ZEROS)
                    .input(2)
                    .addLayer(2, ActivationFunctionType.SIGMOID)
                    .addLayer(2, ActivationFunctionType.SIGMOID)
                    .buildBatchNormModel();

            double[][] firstWeightMatrix = new double[][]{
                    {0.15, 0.2},
                    {0.25, 0.3}
            };
            double[] firstBiasMatrix = new double[]{0.35, 0.35};
            neuralNetworkModel.getLayers().get(0).setWeightMatrix(new DoubleMatrix(firstWeightMatrix));
            neuralNetworkModel.getLayers().get(0).setGammaMatrix(new DoubleMatrix(firstBiasMatrix));
            neuralNetworkModel.getLayers().get(0).setBetaMatrix(new DoubleMatrix(firstBiasMatrix));

            double[][] secondWeightMatrix = new double[][]{
                    {0.40, 0.45},
                    {0.50, 0.55}
            };
            double[] secondBiasMatrix = new double[]{0.6, 0.6};
            neuralNetworkModel.getLayers().get(1).setWeightMatrix(new DoubleMatrix(secondWeightMatrix));
            neuralNetworkModel.getLayers().get(1).setGammaMatrix(new DoubleMatrix(secondBiasMatrix));
            neuralNetworkModel.getLayers().get(1).setBetaMatrix(new DoubleMatrix(secondBiasMatrix));

            double[] input = new double[]{0.05, 0.1};

            double[] output = new double[]{0.01, 0.99};

            double[][] expectedFirstWeightMatrix = {
                    {0.149781, 0.249751},
                    {0.199561, 0.299502}
            };
            double[] expectedFirstBiasMatrix = {0.345614, 0.345023};

            double[][] expectedSecondWeightMatrix = {
                    {0.358916, 0.511301},
                    {0.408666, 0.561370}
            };
            double[] expectedSecondBiasMatrix = {0.530751, 0.619049};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            gradientDescent.updateLearningRate(0.5);
            NeuralNetworkModel<BatchNormLayer> gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            gradientDescentWithDerivation.updateLearningRate(0.5);
            NeuralNetworkModel<BatchNormLayer> gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

    private IGradientDescentProcessProvider getGradientDescentProvider() {
        return new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentBatchNormProcessProvider(new GradientDescentDefaultProcessProvider()));
    }
}
