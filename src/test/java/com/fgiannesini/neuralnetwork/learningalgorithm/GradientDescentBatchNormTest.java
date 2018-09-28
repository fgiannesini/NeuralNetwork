package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnSoftMaxRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class GradientDescentBatchNormTest {

    private IGradientDescentProcessProvider getGradientDescentProvider() {
        return new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider());
    }

    @Nested
    class VariationOnInputAndLayerSize {

        @Test
        void learn_on_vector_with_one_hidden_layer() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addBatchNormLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new BatchNormData(new DoubleMatrix(2, 1, 1, 2), new MeanDeviationProvider());
            LayerTypeData output = new BatchNormData(new DoubleMatrix(2, 1, 3, 5), null);

            double[][] expectedWeightMatrix = {
                    {1, 1},
                    {1, 1}
            };
            double[] expectedGammaMatrix = {1, 1};

            double[] expectedBetaMatrix = {1.02, 1.04};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_one_hidden_layer_with_softmax_layer_and_random_input() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(2)
                    .addBatchNormLayer(3, ActivationFunctionType.NONE)
                    .addBatchNormLayer(2, ActivationFunctionType.SOFT_MAX)
                    .buildNeuralNetworkModel();

            BatchNormData input = new BatchNormData(DoubleMatrix.rand(2, 3), new MeanDeviationProvider());
            BatchNormData output = new BatchNormData(DoubleMatrix.zeros(2, 3), null);
            output.getInput().putRow(0, DoubleMatrix.ones(1, 3).subi(input.getInput()));
            output.getInput().putRow(1, input.getInput());

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnSoftMaxRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.SOFT_MAX_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);

            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_one_hidden_layer() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addBatchNormLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new BatchNormData(new DoubleMatrix(2, 2, 1, 2, 3, 4), new MeanDeviationProvider());
            LayerTypeData output = new BatchNormData(new DoubleMatrix(3, 5, 7, 9), null);

            double[][] expectedWeightMatrix = {
                    {1, 1},
                    {1, 1}
            };
            double[] expectedGammaMatrix = {1.01, 1.01};

            double[] expectedBetaMatrix = {1.04, 1.06};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientWithDerivativeNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedBetaMatrix)));
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }


        @Test
        void learn_on_matrix_with_two_hidden_layers() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addBatchNormLayer(3, ActivationFunctionType.NONE)
                    .addBatchNormLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 3, 2, 4, 5, 6, 9, 10));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 3, 11, 13, 15, 17, 19, 21));

            double[][] expectedFirstWeightMatrix = {
                    {1.00008, 1.00008, 1.00008},
                    {0.99992, 0.999920, 0.99992}
            };
            double[] expectedFirstGammaMatrix = {1, 1, 1};
            double[] expectedFirstBetaMatrix = {1, 1, 1};

            double[][] expectedSecondWeightMatrix = {
                    {1, 1},
                    {1, 1},
                    {1, 1}
            };
            double[] expectedSecondGammaMatrix = {1.02237, 1.02237};
            double[] expectedSecondBetaMatrix = {1.14, 1.16};

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, getGradientDescentProvider());
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 1, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedSecondWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedSecondBetaMatrix)));
            NeuralNetworkAssertions.checkNeuralNetworksLayer(gradientNeuralNetworkModel, 0, Arrays.asList(DataFormatConverter.fromDoubleTabToDoubleMatrix(expectedFirstWeightMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstGammaMatrix), DataFormatConverter.fromTabToDoubleMatrix(expectedFirstBetaMatrix)));

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}
