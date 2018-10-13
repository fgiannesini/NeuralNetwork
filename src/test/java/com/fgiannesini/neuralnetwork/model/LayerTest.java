package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LayerTest {

    @Test
    void test_weightBiasLayer_clone() {
        WeightBiasLayer layer = new WeightBiasLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        WeightBiasLayer clone = layer.clone();

        Assertions.assertNotSame(clone, layer);

        Assertions.assertEquals(clone.getBiasMatrix(), layer.getBiasMatrix());
        Assertions.assertNotSame(clone.getBiasMatrix(), layer.getBiasMatrix());

        Assertions.assertEquals(clone.getWeightMatrix(), layer.getWeightMatrix());
        Assertions.assertNotSame(clone.getWeightMatrix(), layer.getWeightMatrix());

        Assertions.assertEquals(clone.getInputLayerSize(), layer.getInputLayerSize());
        Assertions.assertEquals(clone.getOutputLayerSize(), layer.getOutputLayerSize());

        Assertions.assertEquals(clone.getActivationFunctionType(), layer.getActivationFunctionType());
    }

    @Test
    void test_batchNormLayer_clone() {
        BatchNormLayer layer = new BatchNormLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        BatchNormLayer clone = layer.clone();

        Assertions.assertNotSame(clone, layer);

        Assertions.assertEquals(clone.getBetaMatrix(), layer.getBetaMatrix());
        Assertions.assertNotSame(clone.getBetaMatrix(), layer.getBetaMatrix());

        Assertions.assertEquals(clone.getGammaMatrix(), layer.getGammaMatrix());
        Assertions.assertNotSame(clone.getGammaMatrix(), layer.getGammaMatrix());

        Assertions.assertEquals(clone.getWeightMatrix(), layer.getWeightMatrix());
        Assertions.assertNotSame(clone.getWeightMatrix(), layer.getWeightMatrix());

        Assertions.assertEquals(clone.getInputLayerSize(), layer.getInputLayerSize());
        Assertions.assertEquals(clone.getOutputLayerSize(), layer.getOutputLayerSize());

        Assertions.assertEquals(clone.getActivationFunctionType(), layer.getActivationFunctionType());
    }

    @Test
    void test_convolutionLayer_clone() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 1, 2, 3, 3, 4, 10, 10, 8, 8);
        ConvolutionLayer clone = layer.clone();

        Assertions.assertNotSame(clone, layer);

        Assertions.assertEquals(clone.getBiasMatrices(), layer.getBiasMatrices());
        Assertions.assertNotSame(clone.getBiasMatrices(), layer.getBiasMatrices());

        Assertions.assertEquals(clone.getWeightMatrices(), layer.getWeightMatrices());
        Assertions.assertNotSame(clone.getWeightMatrices(), layer.getWeightMatrices());

        Assertions.assertEquals(clone.getOutputChannelCount(), layer.getOutputChannelCount());
        Assertions.assertEquals(clone.getFilterSize(), layer.getFilterSize());
        Assertions.assertEquals(clone.getPadding(), layer.getPadding());
        Assertions.assertEquals(clone.getStride(), layer.getStride());

        Assertions.assertEquals(clone.getActivationFunctionType(), layer.getActivationFunctionType());
    }

    @Test
    void test_MaxPoolingLayer_clone() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 1, 2, 3, 4, 10, 10, 8, 8);
        MaxPoolingLayer clone = layer.clone();

        Assertions.assertNotSame(clone, layer);

        Assertions.assertEquals(clone.getFilterSize(), layer.getFilterSize());
        Assertions.assertEquals(clone.getPadding(), layer.getPadding());
        Assertions.assertEquals(clone.getStride(), layer.getStride());

        Assertions.assertEquals(clone.getActivationFunctionType(), layer.getActivationFunctionType());
    }

    @Test
    void test_AveragePoolingLayer_clone() {
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 1, 2, 3, 4, 10, 10, 8, 8);
        AveragePoolingLayer clone = layer.clone();

        Assertions.assertNotSame(clone, layer);

        Assertions.assertEquals(clone.getFilterSize(), layer.getFilterSize());
        Assertions.assertEquals(clone.getPadding(), layer.getPadding());
        Assertions.assertEquals(clone.getStride(), layer.getStride());

        Assertions.assertEquals(clone.getActivationFunctionType(), layer.getActivationFunctionType());
    }
}