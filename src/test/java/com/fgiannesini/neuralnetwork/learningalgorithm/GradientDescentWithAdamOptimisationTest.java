package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentWithAdamOptimisationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentWithAdamOptimisationProcessProvider(momentumCoeff, rmsStopCoeff));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            /*LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, new GradientDescentWithDerivationAndRmsStopProcessProvider(rmsStopCoeff));
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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentWithAdamOptimisationProcessProvider(momentumCoeff, rmsStopCoeff));
            gradientDescent.learn(input, output);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            /*LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, new GradientDescentWithDerivationAndRmsStopProcessProvider(rmsStopCoeff));
            gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
            */
        }
    }

}