package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentDefaultProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.GradientDescentOnLogisticRegressionProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.GradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class GradientDescentOnLogisticRegressionTest {

    @Nested
    class VariationOnInputAndLayerSize {
        @Test
        void learn_on_vector_with_one_hidden_layer() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(2, ActivationFunctionType.SIGMOID)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 3, 4));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 1, 0));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLogisticRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LOGISTIC_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_vector_with_two_hidden_layers() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.SIGMOID)
                    .addWeightBiasLayer(2, ActivationFunctionType.SIGMOID)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 3, 4));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 1, 0));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLogisticRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);

            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LOGISTIC_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }

        @Test
        void learn_on_matrix_with_two_hidden_layers() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.ONES)
                    .input(2)
                    .addWeightBiasLayer(3, ActivationFunctionType.SIGMOID)
                    .addWeightBiasLayer(2, ActivationFunctionType.SIGMOID)
                    .buildNeuralNetworkModel();

            LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 2, 1, 2, 3, 4));
            LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 1, 0, 0, 1));

            LearningAlgorithm gradientDescent = new GradientDescent(neuralNetworkModel, new GradientDescentOnLogisticRegressionProcessProvider(new GradientDescentDefaultProcessProvider()));
            NeuralNetworkModel gradientNeuralNetworkModel = gradientDescent.learn(input, output);
            LearningAlgorithm gradientDescentWithDerivation = new GradientDescentWithDerivation(neuralNetworkModel, CostType.LOGISTIC_REGRESSION, new GradientDescentWithDerivationProcessProvider());
            NeuralNetworkModel gradientWithDerivativeNeuralNetworkModel = gradientDescentWithDerivation.learn(input, output);
            NeuralNetworkAssertions.checkSameNeuralNetworks(gradientNeuralNetworkModel, gradientWithDerivativeNeuralNetworkModel);
        }
    }

}