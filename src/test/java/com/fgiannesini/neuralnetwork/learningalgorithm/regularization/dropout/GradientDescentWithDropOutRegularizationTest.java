package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class GradientDescentWithDropOutRegularizationTest {

    @Test
    void learn_on_vector_with_one_hidden_layer() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();

        double[] input = new double[]{1, 2};
        double[] output = new double[]{3, 5};

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1}),
                new DoubleMatrix(new double[]{0, 2})
        );
        LearningAlgorithm gradientDescent = new GradientDescentWithDropOutRegularization(neuralNetworkModel, 0.01, dropOutMatrices);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivationAndDropOutRegularization(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, dropOutMatrices);
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

        double[] output = new double[]{15, 15};

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1}),
                new DoubleMatrix(new double[]{1.5, 0, 1.5}),
                new DoubleMatrix(new double[]{0, 2})
        );
        LearningAlgorithm gradientDescent = new GradientDescentWithDropOutRegularization(neuralNetworkModel, 0.01, dropOutMatrices);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivationAndDropOutRegularization(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, dropOutMatrices);
        NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
        NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
    }

    @Test
    void learn_on_vector_with_two_hidden_layers_with_tanh_activation_function() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(3, ActivationFunctionType.TANH)
                .addLayer(2, ActivationFunctionType.TANH)
                .build();

        double[] input = new double[]{1, 2};

        double[] output = new double[]{15, 15};

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1}),
                new DoubleMatrix(new double[]{1.5, 0, 1.5}),
                new DoubleMatrix(new double[]{0, 2})
        );

        LearningAlgorithm gradientDescent = new GradientDescentWithDropOutRegularization(neuralNetworkModel, 0.01, dropOutMatrices);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivationAndDropOutRegularization(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, dropOutMatrices);
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

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1}),
                new DoubleMatrix(new double[]{1.5, 0, 1.5}),
                new DoubleMatrix(new double[]{0, 2})
        );
        LearningAlgorithm gradientDescent = new GradientDescentWithDropOutRegularization(neuralNetworkModel, 0.01, dropOutMatrices);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivationAndDropOutRegularization(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, dropOutMatrices);
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

        List<DoubleMatrix> dropOutMatrices = DropOutUtils.init()
                .getDropOutMatrix(new double[]{1, 2d / 3d, 0.5}, neuralNetworkModel.getLayers());

        LearningAlgorithm gradientDescent = new GradientDescentWithDropOutRegularization(neuralNetworkModel, 0.01, dropOutMatrices);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivationAndDropOutRegularization(neuralNetworkModel, CostType.LINEAR_REGRESSION, 0.01, dropOutMatrices);
        NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
        NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
    }
}