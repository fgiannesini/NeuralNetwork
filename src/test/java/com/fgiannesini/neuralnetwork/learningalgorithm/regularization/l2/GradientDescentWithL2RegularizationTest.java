package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Test;

class GradientDescentWithL2RegularizationTest {

    @Test
    void learn_on_matrix_with_two_hidden_layers() {
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

        LearningAlgorithm gradientDescent = new GradientDescentWithL2Regularization(neuralNetworkModel, 0.01, 0.5);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivationAndL2Regularization(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, 0.5);
        NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
        NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
    }

    @Test
    void learn_on_matrix_with_two_hidden_layers_and_random_weights() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(2)
                .addLayer(3, ActivationFunctionType.TANH)
                .addLayer(2, ActivationFunctionType.TANH)
                .build();

        double[][] input = new double[][]{
                {1, 2},
                {3, 4}
        };

        double[][] output = new double[][]{
                {15, 15},
                {20, 20}
        };

        LearningAlgorithm gradientDescent = new GradientDescentWithL2Regularization(neuralNetworkModel, 0.01, 0.5);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivationAndL2Regularization(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, 0.5);
        NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
        NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
    }
}