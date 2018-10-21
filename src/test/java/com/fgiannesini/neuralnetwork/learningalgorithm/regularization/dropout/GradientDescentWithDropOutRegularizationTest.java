package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.IGradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

class GradientDescentWithDropOutRegularizationTest {

    @Nested
    class WeightBiasLayer {
        @Test
        void learn_on_vector_with_one_hidden_layer() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 1, 2));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 3, 5));

            Supplier<List<DoubleMatrix>> dropOutMatrices = () -> Arrays.asList(
                    new DoubleMatrix(new double[]{1, 1}),
                    new DoubleMatrix(new double[]{0, 2})
            );

            IGradientDescentProcessProvider processProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentWithDerivationProcessProvider());
            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, withDerivationProcessProvider);
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
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 15, 15));

            Supplier<List<DoubleMatrix>> dropOutMatrices = () -> Arrays.asList(
                    new DoubleMatrix(new double[]{1, 1}),
                    new DoubleMatrix(new double[]{1.5, 0, 1.5}),
                    new DoubleMatrix(new double[]{0, 2})
            );

            IGradientDescentProcessProvider processProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentWithDerivationProcessProvider());
            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, withDerivationProcessProvider);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_two_hidden_layers_with_tanh_activation_function() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.TANH)
                    .addWeightBiasLayer(2, ActivationFunctionType.TANH)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 1, 2));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 15, 15));

            Supplier<List<DoubleMatrix>> dropOutMatrices = () -> Arrays.asList(
                    new DoubleMatrix(new double[]{1, 1}),
                    new DoubleMatrix(new double[]{1.5, 0, 1.5}),
                    new DoubleMatrix(new double[]{0, 2})
            );

            IGradientDescentProcessProvider processProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentWithDerivationProcessProvider());
            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, withDerivationProcessProvider);
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

            Supplier<List<DoubleMatrix>> dropOutMatrices = () -> Arrays.asList(
                    new DoubleMatrix(new double[]{1, 1}),
                    new DoubleMatrix(new double[]{1.5, 0, 1.5}),
                    new DoubleMatrix(new double[]{0, 2})
            );

            IGradientDescentProcessProvider processProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentWithDerivationProcessProvider());
            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, withDerivationProcessProvider);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers_and_random_weights() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.TANH)
                    .addWeightBiasLayer(2, ActivationFunctionType.TANH)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 2, 1, 2, 3, 4));

            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 15, 15, 20, 20));

            List<DoubleMatrix> dropOutMatrix = DropOutUtils.init()
                    .getDropOutMatrix(new double[]{1, 2d / 3d, 0.5}, neuralNetworkModel.getLayers());
            Supplier<List<DoubleMatrix>> dropOutMatrices = () -> dropOutMatrix;

            IGradientDescentProcessProvider processProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentWithDerivationProcessProvider());
            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, withDerivationProcessProvider);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

    @Nested
    class ConvolutionLayer {

        @Test
        void learn_on_matrix_with_two_hidden_layers_and_random_weights() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .input(7, 7, 2)
                    .addConvolutionLayer(3, 0, 1, 2, ActivationFunctionType.TANH)
                    .addAveragePoolingLayer(2, 0, 1, ActivationFunctionType.TANH)
                    .addFullyConnectedLayer(2, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel();

            LayerTypeData input = new ConvolutionData(Arrays.asList(
                    DoubleMatrix.rand(7, 7),
                    DoubleMatrix.rand(7, 7),

                    DoubleMatrix.rand(7, 7),
                    DoubleMatrix.rand(7, 7)
            ), 2);

            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 15, 15, 20, 20));

            List<DoubleMatrix> dropOutMatrix = DropOutUtils.init()
                    .getDropOutMatrix(new double[]{1, 2d / 3d, 1, 0.5}, neuralNetworkModel.getLayers());
            Supplier<List<DoubleMatrix>> dropOutMatrices = () -> dropOutMatrix;

            IGradientDescentProcessProvider processProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(dropOutMatrices, new GradientDescentWithDerivationProcessProvider());
            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, withDerivationProcessProvider);
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }
}