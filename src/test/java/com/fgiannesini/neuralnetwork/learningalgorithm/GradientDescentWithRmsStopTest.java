package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentWithRmsStopProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationAndRmsStopProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class GradientDescentWithRmsStopTest {

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
                    {0.68377, 0.68377, 0.68377},
                    {0.68377, 0.68377, 0.68377}
            };
            double[] expectedFirstBiasMatrix = {0.68377, 0.68377, 0.68377};

            double[][] expectedSecondWeightMatrix = {
                    {0.68377, 0.68377},
                    {0.68377, 0.68377},
                    {0.68377, 0.68377}
            };
            double[] expectedSecondBiasMatrix = {0.68377, 0.68377};
            double rmsStopCoeff = 0.999;

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithRmsStopProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(), rmsStopCoeff));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndRmsStopProcessProvider(new GradientDescentWithDerivationProcessProvider(), rmsStopCoeff));
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
                    {0.95779, 0.95779, 0.95779},
                    {0.96999, 0.96999, 0.96999}
            };
            double[] expectedFirstBiasMatrix = {0.98974, 0.98974, 0.98974};

            double[][] expectedSecondWeightMatrix = {
                    {0.96999, 0.96999},
                    {0.96999, 0.96999},
                    {0.96999, 0.96999}
            };
            double[] expectedSecondBiasMatrix = {0.99508, 0.99508};
            double rmsStopCoeff = 0.999;

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithRmsStopProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(), rmsStopCoeff));
            gradientDescent.learn(input, output);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndRmsStopProcessProvider(new GradientDescentWithDerivationProcessProvider(), rmsStopCoeff));
            gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}