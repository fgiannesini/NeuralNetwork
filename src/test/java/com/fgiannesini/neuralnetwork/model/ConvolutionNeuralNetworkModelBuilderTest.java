package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

class ConvolutionNeuralNetworkModelBuilderTest {

    @Nested
    class ConvolutionNeuralNetworkModel {

        @Test
        void create_Network_Model_Missing_InputSize_Throw_Exception() {
            Assertions.assertThrows(IllegalArgumentException.class, () -> ConvolutionNeuralNetworkModelBuilder.init()
                    .addConvolutionLayer(5, 0, 0, 1, ActivationFunctionType.NONE)
                    .addAveragePoolingLayer(3, 0, 0, ActivationFunctionType.NONE)
                    .addFullyConnectedLayer(10, ActivationFunctionType.NONE)
                    .buildConvolutionNetworkModel());
        }

        @Test
        void create_Network_Model_Missing_Layer_Throw_Exception() {
            Assertions.assertThrows(IllegalArgumentException.class, () -> ConvolutionNeuralNetworkModelBuilder.init()
                    .input(10, 10, 3)
                    .buildConvolutionNetworkModel());
        }

        @Test
        void create_Network_Model() {
            NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                    .input(34, 34, 3)
                    .addConvolutionLayer(3, 1, 3, 4, ActivationFunctionType.NONE)
                    .addAveragePoolingLayer(5, 2, 1, ActivationFunctionType.NONE)
                    .addMaxPoolingLayer(3, 1, 1, ActivationFunctionType.NONE)
                    .addFullyConnectedLayer(10, ActivationFunctionType.SOFT_MAX)
                    .buildConvolutionNetworkModel();

            List<Layer> layers = neuralNetworkModel.getLayers();
            Assertions.assertEquals(4, layers.size());

            ConvolutionLayer firstLayer = (ConvolutionLayer) layers.get(0);

            List<DoubleMatrix> firstWeightMatrices = firstLayer.getWeightMatrices();
            Assertions.assertEquals(12, firstWeightMatrices.size());
            firstWeightMatrices.forEach(w -> Assertions.assertEquals(3, w.getRows()));
            firstWeightMatrices.forEach(w -> Assertions.assertEquals(3, w.getColumns()));

            List<DoubleMatrix> firstBiasMatrices = firstLayer.getBiasMatrices();
            Assertions.assertEquals(4, firstBiasMatrices.size());
            firstBiasMatrices.forEach(w -> Assertions.assertEquals(1, w.getRows()));
            firstBiasMatrices.forEach(w -> Assertions.assertEquals(1, w.getColumns()));

            Assertions.assertEquals(4, firstLayer.getOutputChannelCount());
            Assertions.assertEquals(3, firstLayer.getFilterSize());
            Assertions.assertEquals(1, firstLayer.getPadding());
            Assertions.assertEquals(3, firstLayer.getStride());
            Assertions.assertEquals(ActivationFunctionType.NONE, firstLayer.getActivationFunctionType());

            AveragePoolingLayer secondLayer = (AveragePoolingLayer) layers.get(1);

            Assertions.assertEquals(5, secondLayer.getFilterSize());
            Assertions.assertEquals(2, secondLayer.getPadding());
            Assertions.assertEquals(1, secondLayer.getStride());
            Assertions.assertEquals(ActivationFunctionType.NONE, secondLayer.getActivationFunctionType());

            MaxPoolingLayer thirdLayer = (MaxPoolingLayer) layers.get(2);

            Assertions.assertEquals(3, thirdLayer.getFilterSize());
            Assertions.assertEquals(1, thirdLayer.getPadding());
            Assertions.assertEquals(1, thirdLayer.getStride());
            Assertions.assertEquals(ActivationFunctionType.NONE, thirdLayer.getActivationFunctionType());

            WeightBiasLayer forthLayer = (WeightBiasLayer) layers.get(3);

            Assertions.assertEquals(10, forthLayer.getWeightMatrix().rows);
            Assertions.assertEquals(576, forthLayer.getWeightMatrix().columns);
            Assertions.assertEquals(10, forthLayer.getBiasMatrix().rows);
            Assertions.assertEquals(1, forthLayer.getBiasMatrix().columns);
            Assertions.assertEquals(ActivationFunctionType.SOFT_MAX, forthLayer.getActivationFunctionType());

        }
    }
}