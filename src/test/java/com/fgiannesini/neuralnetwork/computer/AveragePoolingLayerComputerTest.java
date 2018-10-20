package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.data.AveragePoolingData;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.model.AveragePoolingLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class AveragePoolingLayerComputerTest {

    @Test
    void convolution_layer_no_padding_no_stride_one_channel_one_input() {
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 3, 0, 1, 1, 5, 5, 3, 3);
        AveragePoolingData input = new AveragePoolingData(Collections.singletonList(DoubleMatrix.ones(5, 5)), 1);
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((AveragePoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(3, 3)));
    }

    @Test
    void convolution_layer_padding_no_stride_one_channel_one_input() {
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 3, 1, 1, 1, 5, 5, 5, 5);
        AveragePoolingData input = new AveragePoolingData(Collections.singletonList(DoubleMatrix.ones(5, 5)), 1);
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((AveragePoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DataFormatConverter.fromDoubleTabToDoubleMatrix(new double[][]{
                {0.44444, 0.66666, 0.66666, 0.66666, 0.44444},
                {0.66666, 1, 1, 1, 0.66666},
                {0.66666, 1, 1, 1, 0.66666},
                {0.66666, 1, 1, 1, 0.66666},
                {0.44444, 0.66666, 0.66666, 0.66666, 0.44444}
        })));
    }

    @Test
    void convolution_layer_no_padding_stride_one_channel_one_input() {
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 3, 0, 2, 1, 11, 11, 5, 5);
        AveragePoolingData input = new AveragePoolingData(Collections.singletonList(DoubleMatrix.ones(11, 11)), 1);
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((AveragePoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(5, 5)));
    }

    @Test
    void convolution_layer_no_padding_no_stride_three_channels_two_inputs() {
        AveragePoolingLayer layer = new AveragePoolingLayer(ActivationFunctionType.NONE, 3, 0, 1, 1, 5, 5, 3, 3);
        AveragePoolingData input = new AveragePoolingData(Arrays.asList(
                DoubleMatrix.ones(5, 5),
                DoubleMatrix.ones(5, 5).muli(2),
                DoubleMatrix.ones(5, 5).muli(3),
                DoubleMatrix.ones(5, 5).muli(4),
                DoubleMatrix.ones(5, 5).muli(5),
                DoubleMatrix.ones(5, 5).muli(6)
        ), 1);

        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        List<DoubleMatrix> expectedMatrix = Arrays.asList(
                DoubleMatrix.ones(3, 3),
                DoubleMatrix.ones(3, 3).muli(2),
                DoubleMatrix.ones(3, 3).muli(3),
                DoubleMatrix.ones(3, 3).muli(4),
                DoubleMatrix.ones(3, 3).muli(5),
                DoubleMatrix.ones(3, 3).muli(6)
        );
        DoubleMatrixAssertions.assertMatrices(((AveragePoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), expectedMatrix);
    }
}