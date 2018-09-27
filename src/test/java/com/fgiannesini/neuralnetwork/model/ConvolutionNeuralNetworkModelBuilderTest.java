package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

class ConvolutionNeuralNetworkModelBuilderTest {

    @Nested
    class WeightBiasNeuralNetworkModel {
        @Test
        void create_Network_Model_Missing_InputSize_Throw_Exception() {
            Assertions.assertThrows(IllegalArgumentException.class, () -> NeuralNetworkModelBuilder.init()
                    .addWeightBiasLayer(5, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(7, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(10, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel());

            Assertions.fail();
        }

        @Test
        void create_Network_Model_Missing_Layer_Throw_Exception() {
            Assertions.assertThrows(IllegalArgumentException.class, () -> NeuralNetworkModelBuilder.init()
                    .input(10)
                    .buildNeuralNetworkModel());
        }

        @Test
        void create_Network_Model() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .input(10)
                    .addWeightBiasLayer(5, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(7, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(10, ActivationFunctionType.SIGMOID)
                    .buildNeuralNetworkModel();

            List<Layer> layers = neuralNetworkModel.getLayers();
            Assertions.assertEquals(3, layers.size());

            WeightBiasLayer firstLayer = (WeightBiasLayer) layers.get(0);
            Assertions.assertAll(
                    () -> Assertions.assertEquals(5, firstLayer.getWeightMatrix().rows),
                    () -> Assertions.assertEquals(10, firstLayer.getWeightMatrix().columns),
                    () -> Assertions.assertEquals(5, firstLayer.getBiasMatrix().rows),
                    () -> Assertions.assertEquals(1, firstLayer.getBiasMatrix().columns),
                    () -> Assertions.assertEquals(ActivationFunctionType.NONE, firstLayer.getActivationFunctionType())
            );

            WeightBiasLayer secondLayer = (WeightBiasLayer) layers.get(1);
            Assertions.assertAll(
                    () -> Assertions.assertEquals(7, secondLayer.getWeightMatrix().rows),
                    () -> Assertions.assertEquals(5, secondLayer.getWeightMatrix().columns),
                    () -> Assertions.assertEquals(7, secondLayer.getBiasMatrix().rows),
                    () -> Assertions.assertEquals(1, secondLayer.getBiasMatrix().columns),
                    () -> Assertions.assertEquals(ActivationFunctionType.NONE, secondLayer.getActivationFunctionType())
            );

            WeightBiasLayer thirdLayer = (WeightBiasLayer) layers.get(2);
            Assertions.assertAll(
                    () -> Assertions.assertEquals(10, thirdLayer.getWeightMatrix().rows),
                    () -> Assertions.assertEquals(7, thirdLayer.getWeightMatrix().columns),
                    () -> Assertions.assertEquals(10, thirdLayer.getBiasMatrix().rows),
                    () -> Assertions.assertEquals(1, thirdLayer.getBiasMatrix().columns)
            );

        }
    }
}