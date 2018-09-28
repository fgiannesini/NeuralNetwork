package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
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

        @Test
        void launch_exception_if_no_HyperParameters() {
            Assertions.assertThrows(
                    IllegalArgumentException.class,
                    () -> {
                        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                                .input(1)
                                .addWeightBiasLayer(1, ActivationFunctionType.RELU)
                                .buildNeuralNetworkModel();
                        return NeuralNetworkBuilder.init()
                                .withNeuralNetworkModel(neuralNetworkModel)
                                .build();
                    }
            );
        }
    }

    @Nested
    class InstanceCreation {

        @Test
        void create_neuralNetwork() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .input(1)
                    .addWeightBiasLayer(1, ActivationFunctionType.RELU)
                    .buildNeuralNetworkModel();
            NeuralNetwork neuralNetwork = NeuralNetworkBuilder.init()
                    .withNeuralNetworkModel(neuralNetworkModel)
                    .withHyperParameters(new HyperParameters())
                    .build();
            Assertions.assertTrue(neuralNetwork instanceof NeuralNetwork);
        }
    }
}