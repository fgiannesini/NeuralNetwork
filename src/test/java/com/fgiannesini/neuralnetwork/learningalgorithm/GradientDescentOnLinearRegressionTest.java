package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class GradientDescentOnLinearRegressionTest {

    @Nested
    class learningIsOptimal {
        @Test
        void learn_on_vector_with_one_hidden_layer_learning_is_optimal() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 1, 2));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 4, 4));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, gradientNeuralNetworkModel);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_two_hidden_layers_learning_is_optimal() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 1, 2));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 13, 13));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, gradientNeuralNetworkModel);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers_learning_is_optimal() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 2, 1, 2, 3, 4));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 13, 13, 25, 25));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, gradientNeuralNetworkModel);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

    }

    @Nested
    class VariationOnInputAndLayerSize {
        @Test
        void learn_on_vector_with_one_hidden_layer() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 1, 2));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 3, 5));

            double[][] expectedWeightMatrix = {
                    {0.99, 1.01},
                    {0.98, 1.02}
            };
            double[] expectedBiasMatrix = {0.99, 1.01};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_two_hidden_layers() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 1, 2));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 10, 15));

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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 2, 1, 2, 3, 4));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 15, 15, 20, 20));

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
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

    @Nested
    class DataExpectation {

        @Test
        void learn_on_vector_with_two_hidden_layers_and_sigmoid_activation_function() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ZEROS)
                    .input(2)
                    .addWeightBiasLayer(2, ActivationFunctionType.SIGMOID)
                    .addWeightBiasLayer(2, ActivationFunctionType.SIGMOID)
                    .buildNeuralNetworkModel();

            double[][] firstWeightMatrix = new double[][]{
                    {0.15, 0.2},
                    {0.25, 0.3}
            };
            double[] firstBiasMatrix = new double[]{0.35, 0.35};
            WeightBiasLayer firstLayer = (WeightBiasLayer) neuralNetworkModel.getLayers().get(0);
            firstLayer.setWeightMatrix(new DoubleMatrix(firstWeightMatrix));
            firstLayer.setBiasMatrix(new DoubleMatrix(firstBiasMatrix));

            double[][] secondWeightMatrix = new double[][]{
                    {0.40, 0.45},
                    {0.50, 0.55}
            };
            double[] secondBiasMatrix = new double[]{0.6, 0.6};
            WeightBiasLayer secondLayer = (WeightBiasLayer) neuralNetworkModel.getLayers().get(1);
            secondLayer.setWeightMatrix(new DoubleMatrix(secondWeightMatrix));
            secondLayer.setBiasMatrix(new DoubleMatrix(secondBiasMatrix));

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 0.05, 0.1));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 0.01, 0.99));

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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            gradientDescent.updateLearningRate(0.5);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            gradientDescentWithDerivation.updateLearningRate(0.5);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}