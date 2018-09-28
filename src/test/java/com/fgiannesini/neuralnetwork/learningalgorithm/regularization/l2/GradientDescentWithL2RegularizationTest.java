package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLinearRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

class GradientDescentWithL2RegularizationTest {

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

        GradientDescentWithL2RegularizationProcessProvider processProvider = new GradientDescentWithL2RegularizationProcessProvider(0.5, neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
        LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        GradientDescentWithDerivationAndL2RegularizationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndL2RegularizationProcessProvider(0.5, new GradientDescentWithDerivationProcessProvider());
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

        GradientDescentWithL2RegularizationProcessProvider processProvider = new GradientDescentWithL2RegularizationProcessProvider(0.5, neuralNetworkModel, new GradientDescentOnLinearRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
        LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, processProvider);
        NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

        GradientDescentWithDerivationAndL2RegularizationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationAndL2RegularizationProcessProvider(0.5, new GradientDescentWithDerivationProcessProvider());
        LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LINEAR_REGRESSION, withDerivationProcessProvider);
        NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
        NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
    }
}