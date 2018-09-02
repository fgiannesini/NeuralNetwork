package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentWithAdamOptimisationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationAndAdamOptimisationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class GradientDescentWithAdamOptimisationTest {

    @Nested
    class VariationOnInputAndLayerSize {

        @Test
        void learn_on_matrix_with_two_hidden_layers_one_turn() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .buildWeightBiasModel();

            double[][] input = new double[][]{
                    {1, 2},
                    {3, 4}
            };

            double[][] output = new double[][]{
                    {15, 15},
                    {20, 20}
            };

            double[][] expectedFirstWeightMatrix = {
                    {0.99, 0.99, 0.99},
                    {0.99, 0.99, 0.99}
            };
            double[] expectedFirstBiasMatrix = {0.99, 0.99, 0.99};

            double[][] expectedSecondWeightMatrix = {
                    {0.99, 0.99},
                    {0.99, 0.99},
                    {0.99, 0.99}
            };
            double[] expectedSecondBiasMatrix = {0.99, 0.99};
            double rmsStopCoeff = 0.999;
            double momentumCoeff = 0.9;

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithAdamOptimisationProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()), momentumCoeff, rmsStopCoeff));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndAdamOptimisationProcessProvider(new GradientDescentWithDerivationProcessProvider(), momentumCoeff, rmsStopCoeff));
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers_two_turns() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .buildWeightBiasModel();

            double[][] input = new double[][]{
                    {1, 2},
                    {3, 4}
            };

            double[][] output = new double[][]{
                    {15, 15},
                    {20, 20}
            };

            double[][] expectedFirstWeightMatrix = {
                    {0.97665, 0.97665, 0.97665},
                    {0.97667, 0.97667, 0.97667}
            };
            double[] expectedFirstBiasMatrix = {0.9768, 0.9768, 0.9768};

            double[][] expectedSecondWeightMatrix = {
                    {0.97667, 0.97667},
                    {0.97667, 0.97667},
                    {0.97667, 0.97667}
            };
            double[] expectedSecondBiasMatrix = {0.97678, 0.97678};
            double rmsStopCoeff = 0.999;
            double momentumCoeff = 0.9;

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithAdamOptimisationProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()), momentumCoeff, rmsStopCoeff));
            gradientDescent.learn(input, output);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndAdamOptimisationProcessProvider(new GradientDescentWithDerivationProcessProvider(), momentumCoeff, rmsStopCoeff));
            gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);

        }
    }

}