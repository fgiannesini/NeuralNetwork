package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.data.MaxPoolingData;
import com.fgiannesini.neuralnetwork.model.MaxPoolingLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class MaxPoolingLayerComputerTest {

    @Test
    void convolution_layer_no_padding_no_stride_one_channel_one_input() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 1, 1, 5, 5, 3, 3);
        MaxPoolingData input = new MaxPoolingData(Collections.singletonList(DoubleMatrix.ones(5, 5)), Collections.emptyList(), Collections.emptyList());
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        MaxPoolingData result = (MaxPoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult();
        DoubleMatrixAssertions.assertMatrices(result.getDatas(), Collections.singletonList(DoubleMatrix.ones(3, 3)));
        DoubleMatrixAssertions.assertMatrices(result.getMaxRowIndexes(), Collections.singletonList(new DoubleMatrix(3, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2)));
        DoubleMatrixAssertions.assertMatrices(result.getMaxColumnIndexes(), Collections.singletonList(new DoubleMatrix(3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2)));
    }

    @Test
    void convolution_layer_padding_no_stride_one_channel_one_input() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 1, 1, 1, 5, 5, 3, 3);
        MaxPoolingData input = new MaxPoolingData(Collections.singletonList(DoubleMatrix.ones(5, 5)), Collections.emptyList(), Collections.emptyList());
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        MaxPoolingData result = (MaxPoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult();
        DoubleMatrixAssertions.assertMatrices(result.getDatas(), Collections.singletonList(DoubleMatrix.ones(5, 5)));
    }

    @Test
    void convolution_layer_no_padding_stride_one_channel_one_input() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 2, 1, 15, 15, 7, 7);
        MaxPoolingData input = new MaxPoolingData(Collections.singletonList(DoubleMatrix.ones(15, 15)), Collections.emptyList(), Collections.emptyList());
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((MaxPoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(7, 7)));
    }

    @Test
    void convolution_layer_no_padding_no_stride_three_channels_two_inputs() {
        MaxPoolingLayer layer = new MaxPoolingLayer(ActivationFunctionType.NONE, 3, 0, 1, 1, 10, 10, 8, 8);
        MaxPoolingData input = new MaxPoolingData(
                Arrays.asList(
                DoubleMatrix.ones(5, 5),
                DoubleMatrix.ones(5, 5).muli(2),
                DoubleMatrix.ones(5, 5).muli(3),
                DoubleMatrix.ones(5, 5).muli(4),
                DoubleMatrix.ones(5, 5).muli(5),
                DoubleMatrix.ones(5, 5).muli(6)
                )
                , Collections.emptyList(), Collections.emptyList());

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
        DoubleMatrixAssertions.assertMatrices(((MaxPoolingData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), expectedMatrix);
    }
}