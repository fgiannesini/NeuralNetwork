package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

class LayerComputerHelperTest {

    @Test
    void compute_on_one_weight_and_weight_bias() {
        WeightBiasLayer layer = new WeightBiasLayer(1, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DoubleMatrix.scalar(3f), layer);
        Assertions.assertArrayEquals(new double[]{4}, output.data);
    }

    @Test
    void compute_on_weight_array_and_weight_bias() {
        WeightBiasLayer layer = new WeightBiasLayer(5, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer);
        Assertions.assertArrayEquals(new double[]{16}, output.data);
    }

    @Test
    void compute_on_weight_matrix_and_weight_bias() {
        WeightBiasLayer layer = new WeightBiasLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer);
        Assertions.assertArrayEquals(new double[]{16, 16}, output.data);
    }

    @Test
    void compute_on_weight_matrix_and_batch_norm() {
        BatchNormLayer layer = new BatchNormLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer);
        Assertions.assertArrayEquals(new double[]{1, 1}, output.data);
    }

    @Test
    void compute_on_weight_matrix_and_batch_norm_with_three_inputs() {
        BatchNormLayer layer = new BatchNormLayer(4, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        double[][] input = new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 9},
                {9, 10, 11, 12}
        };

        double[][] expectedOutput = new double[][]{
                {-0.2494, -0.2494},
                {1.0509, 1.0509},
                {2.1984, 2.1984}
        };
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DataFormatConverter.fromDoubleTabToDoubleMatrix(input), layer);
        IntStream.range(0, expectedOutput.length).forEach(i -> Assertions.assertArrayEquals(expectedOutput[i], DataFormatConverter.fromDoubleMatrixToDoubleTab(output)[i],0.0001));
    }

}