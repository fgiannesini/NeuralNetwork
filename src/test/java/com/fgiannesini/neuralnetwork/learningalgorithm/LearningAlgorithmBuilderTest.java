package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LearningAlgorithmBuilderTest {

    @Test
    void test_exception_if_neuralNetworkModel_is_missing() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> LearningAlgorithmBuilder.init().build());
    }

    private NeuralNetworkModel buildNeuralNetworkModel() {
        return NeuralNetworkModelBuilder.init()
                .input(1)
                .addLayer(1)
                .build();
    }

    @Test
    void test_GradientDescent_instance_creation() {
        LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                .withModel(buildNeuralNetworkModel())
                .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                .build();

        Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
    }

    @Test
    void test_GradientDescentWithL2Regularization_instance_creation() {
        LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                .withModel(buildNeuralNetworkModel())
                .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                .withL2Regularization(0.5)
                .build();

        Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithL2Regularization);
    }

    @Test
    void test_GradientDescentWithDerivation_instance_creation() {
        LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                .withModel(buildNeuralNetworkModel())
                .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                .build();

        Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
    }

    @Test
    void test_GradientDescentWithDerivationAndL2Regularization_instance_creation() {
        LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                .withModel(buildNeuralNetworkModel())
                .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                .withL2Regularization(0.5)
                .build();

        Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivationAndL2Regularization);
    }

}