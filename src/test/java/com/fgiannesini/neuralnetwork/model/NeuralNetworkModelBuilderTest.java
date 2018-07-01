package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class NeuralNetworkModelBuilderTest {

    @Test
    void create_Network_Model_Missing_InputSize_Throw_Exception() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> NeuralNetworkModelBuilder.init()
                .addLayer(5)
                .addLayer(7)
                .addLayer(10)
                .build());
    }

    @Test
    void create_Network_Model_Missing_Layer_Throw_Exception() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> NeuralNetworkModelBuilder.init()
                .input(10)
                .build());
    }

    @Test
    void create_Network_Model_Default_Activation_Function() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(10)
                .addLayer(5)
                .addLayer(3)
                .build();

        Assertions.assertAll(
                () -> Assertions.assertEquals(10, neuralNetworkModel.getInputSize()),
                () -> Assertions.assertEquals(3, neuralNetworkModel.getOutputSize())
        );

        List<Layer> layers = neuralNetworkModel.getLayers();
        Assertions.assertEquals(2, layers.size());

        Layer firstLayer = layers.get(0);
        Assertions.assertAll(
                () -> Assertions.assertEquals(10, firstLayer.getWeightMatrix().rows),
                () -> Assertions.assertEquals(5, firstLayer.getWeightMatrix().columns),
                () -> Assertions.assertEquals(1, firstLayer.getBiasMatrix().rows),
                () -> Assertions.assertEquals(5, firstLayer.getBiasMatrix().columns),
                () -> Assertions.assertEquals(ActivationFunctionType.RELU, firstLayer.getActivationFunctionType(), "Default layer activation function is not None")
        );

        Layer secondLayer = layers.get(1);
        Assertions.assertAll(
                () -> Assertions.assertEquals(5, secondLayer.getWeightMatrix().rows),
                () -> Assertions.assertEquals(3, secondLayer.getWeightMatrix().columns),
                () -> Assertions.assertEquals(1, secondLayer.getBiasMatrix().rows),
                () -> Assertions.assertEquals(3, secondLayer.getBiasMatrix().columns),
                () -> Assertions.assertEquals(ActivationFunctionType.RELU, secondLayer.getActivationFunctionType(), "Last layer activation function is not Sigmoid")
        );
    }

    @Test
    void create_Network_Model_Override_Activation_Function() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(10)
                .addLayer(5, ActivationFunctionType.NONE)
                .addLayer(7, ActivationFunctionType.NONE)
                .addLayer(10, ActivationFunctionType.SIGMOID)
                .build();

        Assertions.assertAll(
                () -> Assertions.assertEquals(10, neuralNetworkModel.getInputSize()),
                () -> Assertions.assertEquals(10, neuralNetworkModel.getOutputSize())
        );

        List<Layer> layers = neuralNetworkModel.getLayers();
        Assertions.assertEquals(3, layers.size());

        Layer firstLayer = layers.get(0);
        Assertions.assertAll(
                () -> Assertions.assertEquals(10, firstLayer.getWeightMatrix().rows),
                () -> Assertions.assertEquals(5, firstLayer.getWeightMatrix().columns),
                () -> Assertions.assertEquals(1, firstLayer.getBiasMatrix().rows),
                () -> Assertions.assertEquals(5, firstLayer.getBiasMatrix().columns),
                () -> Assertions.assertEquals(ActivationFunctionType.NONE, firstLayer.getActivationFunctionType())
        );

        Layer secondLayer = layers.get(1);
        Assertions.assertAll(
                () -> Assertions.assertEquals(5, secondLayer.getWeightMatrix().rows),
                () -> Assertions.assertEquals(7, secondLayer.getWeightMatrix().columns),
                () -> Assertions.assertEquals(1, secondLayer.getBiasMatrix().rows),
                () -> Assertions.assertEquals(7, secondLayer.getBiasMatrix().columns),
                () -> Assertions.assertEquals(ActivationFunctionType.NONE, secondLayer.getActivationFunctionType())
        );

        Layer thirdLayer = layers.get(2);
        Assertions.assertAll(
                () -> Assertions.assertEquals(7, thirdLayer.getWeightMatrix().rows),
                () -> Assertions.assertEquals(10, thirdLayer.getWeightMatrix().columns),
                () -> Assertions.assertEquals(1, thirdLayer.getBiasMatrix().rows),
                () -> Assertions.assertEquals(10, thirdLayer.getBiasMatrix().columns),
                () -> Assertions.assertEquals(ActivationFunctionType.SIGMOID, thirdLayer.getActivationFunctionType())
        );
    }

}