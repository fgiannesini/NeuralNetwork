package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.ConvolutionLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

import java.util.Collections;

class ConvolutionLayerComputerTest {

    @Test
    void convolution_layer_no_padding_no_stride_one_channel() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 3, 0, 1, 1, 1);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(5, 5)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DoubleMatrix.ones(3, 3).muli(9)));
    }

    @Test
    void convolution_layer_padding_no_stride_one_channel() {
        ConvolutionLayer layer = new ConvolutionLayer(ActivationFunctionType.NONE, InitializerType.ONES.getInitializer(), 3, 1, 1, 1, 1);
        ConvolutionData input = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(5, 5)));
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        DoubleMatrixAssertions.assertMatrices(((ConvolutionData) layerComputerVisitor.getIntermediateOutputResult().getResult()).getDatas(), Collections.singletonList(DataFormatConverter.fromDoubleTabToDoubleMatrix(new double[][]{
                {4, 6, 6, 6, 4},
                {6, 9, 9, 9, 6},
                {6, 9, 9, 9, 6},
                {6, 9, 9, 9, 6},
                {4, 6, 6, 6, 4}
        })));
    }
}