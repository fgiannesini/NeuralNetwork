package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.AveragePoolingLayer;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.MaxPoolingLayer;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class DataAdapterVisitorTest {

    @Test
    void from_weightBias_to_weightBias() {
        WeightBiasData inputData = new WeightBiasData(DoubleMatrix.EMPTY);
        DataAdapterVisitor dataAdapterVisitor = new DataAdapterVisitor(inputData);
        WeightBiasLayer layer = new WeightBiasLayer(1, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        layer.accept(dataAdapterVisitor);
        Assertions.assertEquals(inputData, dataAdapterVisitor.getData());
    }

    @Test
    void from_batchNorm_to_batchNorm() {
        BatchNormData inputData = new BatchNormData(DoubleMatrix.EMPTY, new MeanDeviationProvider());
        DataAdapterVisitor dataAdapterVisitor = new DataAdapterVisitor(inputData);
        BatchNormLayer layer = new BatchNormLayer(1, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        layer.accept(dataAdapterVisitor);
        Assertions.assertEquals(inputData, dataAdapterVisitor.getData());
    }

    @Test
    void from_convolution_to_averagePooling() {
        ConvolutionData inputData = new ConvolutionData(Collections.singletonList(DoubleMatrix.EMPTY));
        DataAdapterVisitor dataAdapterVisitor = new DataAdapterVisitor(inputData);
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 3, 0, 1);
        layer.accept(dataAdapterVisitor);
        Assertions.assertEquals(inputData, dataAdapterVisitor.getData());
    }

    @Test
    void from_convolution_to_maxPooling() {
        ConvolutionData inputData = new ConvolutionData(Collections.singletonList(DoubleMatrix.EMPTY));
        DataAdapterVisitor dataAdapterVisitor = new DataAdapterVisitor(inputData);
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 1);
        layer.accept(dataAdapterVisitor);
        Assertions.assertEquals(inputData, dataAdapterVisitor.getData());
    }

    @Test
    void from_averagePooling_to_weightBias() {
        List<DoubleMatrix> inputMatrices = Arrays.asList(
                DoubleMatrix.ones(2, 2).muli(1),
                DoubleMatrix.ones(2, 2).muli(2),
                DoubleMatrix.ones(2, 2).muli(11),
                DoubleMatrix.ones(2, 2).muli(12)
        );
        ConvolutionData inputData = new ConvolutionData(inputMatrices);
        DataAdapterVisitor dataAdapterVisitor = new DataAdapterVisitor(inputData);
        WeightBiasLayer layer = new WeightBiasLayer(8, 3, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        layer.accept(dataAdapterVisitor);
        DoubleMatrix expected = new DoubleMatrix(8, 2, 1, 1, 1, 1, 2, 2, 2, 2, 11, 11, 11, 11, 12, 12, 12, 12);
        DoubleMatrixAssertions.assertMatrices(expected, ((WeightBiasData) dataAdapterVisitor.getData()).getInput());
    }

}