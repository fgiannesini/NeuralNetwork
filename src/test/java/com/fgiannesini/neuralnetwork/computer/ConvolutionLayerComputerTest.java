package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.ConvolutionLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class ConvolutionLayerComputerTest {

    @Test
    void convolution_layer_no_padding_no_stride_one_channel_one_input() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 3, 0, 1, 1, 1, 10, 10);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(5, 5)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(3, 3).muli(10)));
    }

    @Test
    void convolution_layer_padding_no_stride_one_channel_one_input() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 3, 1, 1, 1, 1, 10, 10);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(5, 5)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DataFormatConverter.fromDoubleTabToDoubleMatrix(new double[][]{
                {5, 7, 7, 7, 5},
                {7, 10, 10, 10, 7},
                {7, 10, 10, 10, 7},
                {7, 10, 10, 10, 7},
                {5, 7, 7, 7, 5}
        })));
    }

    @Test
    void convolution_layer_no_padding_stride_one_channel_one_input() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 3, 0, 2, 1, 1, 10, 10);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(11, 11)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(5, 5).muli(10)));
    }

    @Test
    void convolution_layer_no_padding_no_stride_three_channels_one_input() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 3, 0, 1, 1, 3, 10, 10);
        ConvolutionData input = new ConvolutionData(Arrays.asList(
                DoubleMatrix.ones(5, 5),
                DoubleMatrix.ones(5, 5).muli(2),
                DoubleMatrix.ones(5, 5).muli(3)
        ));
        layer.getWeightMatrices().get(0).muli(1);
        layer.getWeightMatrices().get(1).muli(2);
        layer.getWeightMatrices().get(2).muli(3);

        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        List<DoubleMatrix> expectedMatrix = Collections.singletonList(DoubleMatrix.ones(3, 3).muli(127));
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), expectedMatrix);
    }

    @Test
    void convolution_layer_no_padding_no_stride_three_channels_two_inputs() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 3, 0, 1, 2, 3, 10, 10);
        ConvolutionData input = new ConvolutionData(Arrays.asList(
                DoubleMatrix.ones(5, 5),
                DoubleMatrix.ones(5, 5).muli(2),
                DoubleMatrix.ones(5, 5).muli(3),
                DoubleMatrix.ones(5, 5).muli(4),
                DoubleMatrix.ones(5, 5).muli(5),
                DoubleMatrix.ones(5, 5).muli(6)
        ));
        layer.getWeightMatrices().get(0).muli(1);
        layer.getWeightMatrices().get(1).muli(2);
        layer.getWeightMatrices().get(2).muli(3);
        layer.getWeightMatrices().get(3).muli(4);
        layer.getWeightMatrices().get(4).muli(5);
        layer.getWeightMatrices().get(5).muli(6);

        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        List<DoubleMatrix> expectedMatrix = Arrays.asList(
                DoubleMatrix.ones(3, 3).muli(127),
                DoubleMatrix.ones(3, 3).muli(289),
                DoubleMatrix.ones(3, 3).muli(289),
                DoubleMatrix.ones(3, 3).muli(694)
        );
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), expectedMatrix);
    }
}