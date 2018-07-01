package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Test;

class GradientDescentWithDerivationTest {

    @Test
    void learn_on_vector_with_one_hidden_layer_learning_is_optimal() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();
        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01f);

        double[] input = new double[]{1, 2};
        double[] output = new double[]{4, 4};

        NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

        NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);
    }

    @Test
    void learn_on_vector_with_two_hidden_layers_learning_is_optimal() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(3, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();
        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01f);

        double[] input = new double[]{1f, 2f};
        double[] output = new double[]{13f, 13f};

        NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
        NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);
    }

    @Test
    void learn_on_matrix_with_two_hidden_layers_learning_is_optimal() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(3, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();
        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01f);

        double[][] input = new double[][]{
                {1, 2},
                {3, 4}
        };

        double[][] output = new double[][]{
                {13, 13},
                {25, 25}
        };

        NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
        NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, optimizedNeuralNetworkModel);
    }

    @Test
    void learn_on_vector_with_one_hidden_layer() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();
        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01f);

        double[] input = new double[]{1, 2};
        double[] output = new double[]{3, 5};

        NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

        double[][] expectedWeightMatrix = {
                {0.99, 1.01},
                {0.98, 1.02}
        };
        double[] expectedBiasMatrix = {0.99, 1.01};

        NeuralNetworkAssertions.checkNeuralNetworksLayer(optimizedNeuralNetworkModel, 0, expectedWeightMatrix, expectedBiasMatrix);
    }

    @Test
    void learn_on_vector_with_two_hidden_layers() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(3, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();
        LearningAlgorithm gradientDescent = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01f);

        double[] input = new double[]{1, 2};
        double[] output = new double[]{10, 15};

        NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescent.learn(input, output);

        double[][] expectedFirstWeightMatrix = {
                {0.96, 0.96, 0.96},
                {0.92, 0.92, 0.92}
        };
        double[] expectedFirstBiasMatrix = {0.96, 0.96, 0.96};
        NeuralNetworkAssertions.checkNeuralNetworksLayer(optimizedNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);

        double[][] expectedSecondWeightMatrix = {
                {0.88, 1.08},
                {0.88, 1.08},
                {0.88, 1.08}
        };
        double[] expectedSecondBiasMatrix = {0.97, 1.02};
        NeuralNetworkAssertions.checkNeuralNetworksLayer(optimizedNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);
    }

    @Test
    void learn_on_matrix_with_two_hidden_layers() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(3, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();
        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01f);

        double[][] input = new double[][]{
                {1, 2},
                {3, 4}
        };

        double[][] output = new double[][]{
                {15, 15},
                {20, 20}
        };

        NeuralNetworkModel optimizedNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

        double[][] expectedFirstWeightMatrix = {
                {-0.12, -0.12, -0.12},
                {-0.44, -0.44, -0.44}
        };
        double[] expectedFirstBiasMatrix = {0.68, 0.68, 0.68};
        NeuralNetworkAssertions.checkNeuralNetworksLayer(optimizedNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);

        double[][] expectedSecondWeightMatrix = {
                {0.84, 0.84},
                {0.84, 0.84},
                {0.84, 0.84}
        };
        double[] expectedSecondBiasMatrix = {0.985, 0.985};
        NeuralNetworkAssertions.checkNeuralNetworksLayer(optimizedNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);
    }
}