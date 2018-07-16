package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class NeuralNetworkBuilderTest {

    @Nested
    class CheckInputs {

        @Test
        void launch_exception_if_no_NeuralNetworkModel() {
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
            NeuralNetwork neuralNetwork = NeuralNetworkBuilder.init()
                    .withNeuralNetworkModel(neuralNetworkModel)
                    .build();
            Assertions.assertTrue(neuralNetwork instanceof NeuralNetwork);
        }
    }
}