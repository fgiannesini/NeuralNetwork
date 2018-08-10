package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentWithMomentumProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class GradientDescentWithMomentumTest {

    @Nested
    class VariationOnInputAndLayerSize {

        @Test
        void learn_on_matrix_with_two_hidden_layers_one_turn() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .build();

            double[][] input = new double[][]{
                    {1, 2},
                    {3, 4}
            };

            double[][] output = new double[][]{
                    {15, 15},
                    {20, 20}
            };

            double[][] expectedFirstWeightMatrix = {
                    {0.987, 0.987, 0.987},
                    {0.984, 0.984, 0.984}
            };
            double[] expectedFirstBiasMatrix = {0.997, 0.997, 0.997};

            double[][] expectedSecondWeightMatrix = {
                    {0.984, 0.984},
                    {0.984, 0.984},
                    {0.984, 0.984}
            };
            double[] expectedSecondBiasMatrix = {0.9985, 0.9985};
            double momentumCoeff = 0.9;

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentWithMomentumProcessProvider(momentumCoeff));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            /*LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
            */
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers_two_turns() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .build();

            double[][] input = new double[][]{
                    {1, 2},
                    {3, 4}
            };

            double[][] output = new double[][]{
                    {15, 15},
                    {20, 20}
            };

            double[][] expectedFirstWeightMatrix = {
                    {0.9764825, 0.9764825, 0.9764825},
                    {0.9715206, 0.9715206, 0.9715206}
            };
            double[] expectedFirstBiasMatrix = {0.9950381, 0.9950381, 0.9950381};

            double[][] expectedSecondWeightMatrix = {
                    {0.971492, 0.971492},
                    {0.971492, 0.971492},
                    {0.971492, 0.971492}
            };
            double[] expectedSecondBiasMatrix = {0.9975033, 0.9975033};
            double momentumCoeff = 0.9;

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentWithMomentumProcessProvider(momentumCoeff));
            gradientDescent.learn(input, output);

            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            /*LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
            */
        }
    }

}