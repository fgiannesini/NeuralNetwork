package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.model.MaxPoolingLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class MaxPoolingLayerComputerTest {

    @Test
    void convolution_layer_no_padding_no_stride_one_channel_one_input() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 1);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(5, 5)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(3, 3)));
    }

    @Test
    void convolution_layer_padding_no_stride_one_channel_one_input() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 1, 1);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(5, 5)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(5, 5)));
    }

    @Test
    void convolution_layer_no_padding_stride_one_channel_one_input() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 2);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(11, 11)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(5, 5)));
    }

    @Test
    void convolution_layer_no_padding_no_stride_three_channels_two_inputs() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 1);
        ConvolutionData input = new ConvolutionData(Arrays.asList(
                DoubleMatrix.ones(5, 5),
                DoubleMatrix.ones(5, 5).muli(2),
                DoubleMatrix.ones(5, 5).muli(3),
                DoubleMatrix.ones(5, 5).muli(4),
                DoubleMatrix.ones(5, 5).muli(5),
                DoubleMatrix.ones(5, 5).muli(6)
        ));

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
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), expectedMatrix);
    }
}