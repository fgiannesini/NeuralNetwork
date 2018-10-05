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
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentWithMomentumProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationAndMomentumProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class GradientDescentWithMomentumTest {

    @Nested
    class VariationOnInputAndLayerSize {

        @Test
        void learn_on_matrix_with_two_hidden_layers_one_turn() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 2, 1, 2, 3, 4));

            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 15, 15, 20, 20));

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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithMomentumProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()), momentumCoeff));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndMomentumProcessProvider(new GradientDescentWithDerivationProcessProvider(), momentumCoeff));
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers_two_turns() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 2, 1, 2, 3, 4));

            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 15, 15, 20, 20));

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

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentWithMomentumProcessProvider(new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()), momentumCoeff));
            gradientDescent.learn(input, output);

            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBiasMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBiasMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationAndMomentumProcessProvider(new GradientDescentWithDerivationProcessProvider(), momentumCoeff));
            gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}