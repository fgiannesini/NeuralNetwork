package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.data.*;
import com.fgiannesini.neuralnetwork.computer.data.adapter.ForwardDataAdapterVisitor;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.*;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class ForwardDataAdapterVisitorTest {

    @Test
    void from_weightBias_to_weightBias() {
        WeightBiasData inputData = new WeightBiasData(DoubleMatrix.EMPTY);
        ForwardDataAdapterVisitor forwardDataAdapterVisitor = new ForwardDataAdapterVisitor(inputData);
        WeightBiasLayer layer = new WeightBiasLayer(1, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        layer.accept(forwardDataAdapterVisitor);
        Assertions.assertEquals(inputData, forwardDataAdapterVisitor.getData());
    }

    @Test
    void from_batchNorm_to_batchNorm() {
        BatchNormData inputData = new BatchNormData(DoubleMatrix.EMPTY, new MeanDeviationProvider());
        ForwardDataAdapterVisitor forwardDataAdapterVisitor = new ForwardDataAdapterVisitor(inputData);
        BatchNormLayer layer = new BatchNormLayer(1, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        layer.accept(forwardDataAdapterVisitor);
        Assertions.assertEquals(inputData, forwardDataAdapterVisitor.getData());
    }

    @Test
    void from_convolution_to_averagePooling() {
        ConvolutionData inputData = new ConvolutionData(Collections.singletonList(DoubleMatrix.rand(10, 10)), 1);
        ForwardDataAdapterVisitor forwardDataAdapterVisitor = new ForwardDataAdapterVisitor(inputData);
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 3, 0, 1, 1, 10, 10, 8, 8);
        layer.accept(forwardDataAdapterVisitor);
        DoubleMatrixAssertions.assertMatrices(inputData.getDatas(), ((AveragePoolingData) forwardDataAdapterVisitor.getData()).getDatas());
    }

    @Test
    void from_convolution_to_maxPooling() {
        ConvolutionData inputData = new ConvolutionData(Collections.singletonList(DoubleMatrix.rand(10, 10)), 1);
        ForwardDataAdapterVisitor forwardDataAdapterVisitor = new ForwardDataAdapterVisitor(inputData);
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 1, 1, 10, 10, 8, 8);
        layer.accept(forwardDataAdapterVisitor);
        DoubleMatrixAssertions.assertMatrices(inputData.getDatas(), ((MaxPoolingData) forwardDataAdapterVisitor.getData()).getDatas());
    }

    @Test
    void from_averagePooling_to_weightBias() {
        List<DoubleMatrix> inputMatrices = getExpectedData();
        ConvolutionData inputData = new ConvolutionData(inputMatrices, 1);
        ForwardDataAdapterVisitor forwardDataAdapterVisitor = new ForwardDataAdapterVisitor(inputData);
        WeightBiasLayer layer = new WeightBiasLayer(8, 3, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        layer.accept(forwardDataAdapterVisitor);
        DoubleMatrix expected = new DoubleMatrix(8, 2, 1, 1, 1, 1, 2, 2, 2, 2, 11, 11, 11, 11, 12, 12, 12, 12);
        DoubleMatrixAssertions.assertMatrices(expected, ((WeightBiasData) forwardDataAdapterVisitor.getData()).getData());
    }

    @Test
    void from_weightBias_to_averagePooling() {
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 3, 1, 1, 2, 2, 2, 2, 2);
        AveragePoolingData layerTypeData = (AveragePoolingData) from_weight_bias_to_pooling(layer);
        DoubleMatrixAssertions.assertMatrices(getExpectedData(), layerTypeData.getDatas());
    }

    @Test
    void from_weightBias_to_maxPooling() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 1, 1, 2, 2, 2, 2, 2);
        MaxPoolingData layerTypeData = (MaxPoolingData) from_weight_bias_to_pooling(layer);
        DoubleMatrixAssertions.assertMatrices(getExpectedData(), layerTypeData.getDatas());
    }

    private LayerTypeData from_weight_bias_to_pooling(Layer layer) {
        DoubleMatrix input = new DoubleMatrix(8, 2, 1, 1, 1, 1, 2, 2, 2, 2, 11, 11, 11, 11, 12, 12, 12, 12);
        WeightBiasData inputData = new WeightBiasData(input);
        ForwardDataAdapterVisitor forwardDataAdapterVisitor = new ForwardDataAdapterVisitor(inputData);
        layer.accept(forwardDataAdapterVisitor);

        return forwardDataAdapterVisitor.getData();
    }

    private List<DoubleMatrix> getExpectedData() {
        return Arrays.asList(
                DoubleMatrix.ones(2, 2).muli(1),
                DoubleMatrix.ones(2, 2).muli(2),
                DoubleMatrix.ones(2, 2).muli(11),
                DoubleMatrix.ones(2, 2).muli(12)
        );
    }

}