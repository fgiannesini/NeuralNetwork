package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithmBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class NeuralNetworkBuilderTest {

    @Nested
    class CheckInputs {

        @Test
        void launch_exception_if_no_learningAlgorithm() {
            Assertions.assertThrows(
                    IllegalArgumentException.class,
                    () -> NeuralNetworkBuilder.init().build()
            );
        }
    }

    @Nested
    class InstanceCreation {

        @Test
        void create_neuralNetwork() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .input(1)
                    .addLayer(1)
                    .build();
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(neuralNetworkModel)
                    .build();
            NeuralNetwork neuralNetwork = NeuralNetworkBuilder.init()
                    .withLearningAlgorithm(learningAlgorithm)
                    .build();
            Assertions.assertTrue(neuralNetwork instanceof NeuralNetwork);
        }
    }
}