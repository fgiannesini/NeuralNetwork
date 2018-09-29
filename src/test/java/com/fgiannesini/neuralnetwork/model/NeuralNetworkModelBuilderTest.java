package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;

class NeuralNetworkModelBuilderTest {

    @Nested
    class WeightBiasNeuralNetworkModel {
        @Test
        void create_Network_Model_Missing_InputSize_Throw_Exception() {
            Assertions.assertThrows(IllegalArgumentException.class, () -> NeuralNetworkModelBuilder.init()
                    .addWeightBiasLayer(5, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(7, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(10, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel());
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
                    () -> Assertions.assertEquals(1, thirdLayer.getBiasMatrix().columns),
                    () -> Assertions.assertEquals(ActivationFunctionType.SIGMOID, thirdLayer.getActivationFunctionType())
            );
        }

        @Test
        void check_random_weight_boudaries() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(256)
                    .addWeightBiasLayer(100, ActivationFunctionType.NONE)
                    .addWeightBiasLayer(100, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            Assertions.assertTrue(
                    Stream.concat(
                            neuralNetworkModel.getLayers().stream().map(layer -> ((WeightBiasLayer) layer).getWeightMatrix()),
                            neuralNetworkModel.getLayers().stream().map(layer -> ((WeightBiasLayer) layer).getBiasMatrix())
                    ).flatMapToDouble(m -> Arrays.stream(m.data))
                            .allMatch(d -> d < 0.01 && d > 0)
            );

        }
    }

    @Nested
    class BatchNormNeuralNetworkModel {
        @Test
        void create_Network_Model_Missing_InputSize_Throw_Exception() {
            Assertions.assertThrows(IllegalArgumentException.class, () -> NeuralNetworkModelBuilder.init()
                    .addBatchNormLayer(5, ActivationFunctionType.NONE)
                    .addBatchNormLayer(7, ActivationFunctionType.NONE)
                    .addBatchNormLayer(10, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel());
        }

        @Test
        void create_Network_Model() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .input(10)
                    .addBatchNormLayer(5, ActivationFunctionType.NONE)
                    .addBatchNormLayer(7, ActivationFunctionType.NONE)
                    .addBatchNormLayer(10, ActivationFunctionType.SIGMOID)
                    .buildNeuralNetworkModel();

            List<Layer> layers = neuralNetworkModel.getLayers();
            Assertions.assertEquals(3, layers.size());

            BatchNormLayer firstLayer = (BatchNormLayer) layers.get(0);
            Assertions.assertAll(
                    () -> Assertions.assertEquals(5, firstLayer.getWeightMatrix().rows),
                    () -> Assertions.assertEquals(10, firstLayer.getWeightMatrix().columns),
                    () -> Assertions.assertEquals(5, firstLayer.getGammaMatrix().rows),
                    () -> Assertions.assertEquals(1, firstLayer.getGammaMatrix().columns),
                    () -> Assertions.assertEquals(5, firstLayer.getBetaMatrix().rows),
                    () -> Assertions.assertEquals(1, firstLayer.getBetaMatrix().columns),
                    () -> Assertions.assertEquals(ActivationFunctionType.NONE, firstLayer.getActivationFunctionType())
            );

            BatchNormLayer secondLayer = (BatchNormLayer) layers.get(1);
            Assertions.assertAll(
                    () -> Assertions.assertEquals(7, secondLayer.getWeightMatrix().rows),
                    () -> Assertions.assertEquals(5, secondLayer.getWeightMatrix().columns),
                    () -> Assertions.assertEquals(7, secondLayer.getGammaMatrix().rows),
                    () -> Assertions.assertEquals(1, secondLayer.getGammaMatrix().columns),
                    () -> Assertions.assertEquals(7, secondLayer.getBetaMatrix().rows),
                    () -> Assertions.assertEquals(1, secondLayer.getBetaMatrix().columns),
                    () -> Assertions.assertEquals(ActivationFunctionType.NONE, secondLayer.getActivationFunctionType())
            );

            BatchNormLayer thirdLayer = (BatchNormLayer) layers.get(2);
            Assertions.assertAll(
                    () -> Assertions.assertEquals(10, thirdLayer.getWeightMatrix().rows),
                    () -> Assertions.assertEquals(7, thirdLayer.getWeightMatrix().columns),
                    () -> Assertions.assertEquals(10, thirdLayer.getGammaMatrix().rows),
                    () -> Assertions.assertEquals(1, thirdLayer.getGammaMatrix().columns),
                    () -> Assertions.assertEquals(10, thirdLayer.getBetaMatrix().rows),
                    () -> Assertions.assertEquals(1, thirdLayer.getBetaMatrix().columns),
                    () -> Assertions.assertEquals(ActivationFunctionType.SIGMOID, thirdLayer.getActivationFunctionType())
            );
        }

        @Test
        void check_random_weight_boudaries() {
            NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                    .useInitializer(InitializerType.RANDOM)
                    .input(256)
                    .addBatchNormLayer(100, ActivationFunctionType.NONE)
                    .addBatchNormLayer(100, ActivationFunctionType.NONE)
                    .buildNeuralNetworkModel();

            Assertions.assertTrue(
                    Stream.of(
                            neuralNetworkModel.getLayers().stream().map(layer -> ((BatchNormLayer) layer).getWeightMatrix()),
                            neuralNetworkModel.getLayers().stream().map(layer -> ((BatchNormLayer) layer).getGammaMatrix()),
                            neuralNetworkModel.getLayers().stream().map(layer -> ((BatchNormLayer) layer).getBetaMatrix())
                    ).flatMap(Function.identity())
                            .flatMapToDouble(m -> Arrays.stream(m.data))
                            .allMatch(d -> d < 0.01 && d > 0)
            );
        }
    }
}