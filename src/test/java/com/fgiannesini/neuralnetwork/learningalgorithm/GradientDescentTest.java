package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class GradientDescentTest {

    @Nested
    public class learningIsOptimal {
        @Test
        void learn_on_vector_with_one_hidden_layer_learning_is_optimal() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .build();

            double[] input = new double[]{1, 2};
            double[] output = new double[]{4, 4};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, gradientNeuralNetworkModel);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_two_hidden_layers_learning_is_optimal() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .build();

            double[] input = new double[]{1, 2};
            double[] output = new double[]{13, 13};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, gradientNeuralNetworkModel);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers_learning_is_optimal() {
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
                    {13, 13},
                    {25, 25}
            };

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, gradientNeuralNetworkModel);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

    }

    @Nested
    public class VariationOnInputAndLayerSize {
        @Test
        void learn_on_vector_with_one_hidden_layer() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .build();

            double[] input = new double[]{1, 2};
            double[] output = new double[]{3, 5};

            double[][] expectedWeightMatrix = {
                    {0.99, 1.01},
                    {0.98, 1.02}
            };
            double[] expectedBiasMatrix = {0.99, 1.01};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedWeightMatrix, expectedBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_two_hidden_layers() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addLayer(3, ActivationFunctionType.NONE)
                    .addLayer(2, ActivationFunctionType.NONE)
                    .build();

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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

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
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.01, new GradientDescentProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

    @Nested
    public class DataExpectation {

        @Test
        void learn_on_vector_with_two_hidden_layers_and_sigmoid_activation_function() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ZEROS)
                    .input(2)
                    .addLayer(2, ActivationFunctionType.SIGMOID)
                    .addLayer(2, ActivationFunctionType.SIGMOID)
                    .build();

            double[][] firstWeightMatrix = new double[][]{
                    {0.15, 0.2},
                    {0.25, 0.3}
            };
            double[] firstBiasMatrix = new double[]{0.35, 0.35};
            neuralNetworkModel.getLayers().get(0).setWeightMatrix(new DoubleMatrix(firstWeightMatrix));
            neuralNetworkModel.getLayers().get(0).setBiasMatrix(new DoubleMatrix(firstBiasMatrix));

            double[][] secondWeightMatrix = new double[][]{
                    {0.40, 0.45},
                    {0.50, 0.55}
            };
            double[] secondBiasMatrix = new double[]{0.6, 0.6};
            neuralNetworkModel.getLayers().get(1).setWeightMatrix(new DoubleMatrix(secondWeightMatrix));
            neuralNetworkModel.getLayers().get(1).setBiasMatrix(new DoubleMatrix(secondBiasMatrix));

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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, 0.5, new GradientDescentProcessProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, expectedFirstWeightMatrix, expectedFirstBiasMatrix);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, expectedSecondWeightMatrix, expectedSecondBiasMatrix);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.5);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}