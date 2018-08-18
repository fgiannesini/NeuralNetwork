package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentWithMomentumProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationAndMomentumProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationProcessProvider;
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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithMomentumProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(), momentumCoeff));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndMomentumProcessProvider(new GradientDescentWithDerivationProcessProvider(), momentumCoeff));
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
                    {0.9648995, 0.9648995, 0.9648995},
                    {0.9572646, 0.9572646, 0.9572646}
            };
            double[] expectedFirstBiasMatrix = {0.9923651, 0.9923651, 0.9923651};

            double[][] expectedSecondWeightMatrix = {
                    {0.957236, 0.957236},
                    {0.957236, 0.957236},
                    {0.957236, 0.957236}

            };
            double[] expectedSecondBiasMatrix = {0.9961668, 0.9961668};
            double momentumCoeff = 0.9;

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithMomentumProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(), momentumCoeff));
            gradientDescent.learn(input, output);

            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndMomentumProcessProvider(new GradientDescentWithDerivationProcessProvider(), momentumCoeff));
            gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}