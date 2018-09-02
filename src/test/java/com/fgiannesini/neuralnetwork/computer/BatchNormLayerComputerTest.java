package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

class BatchNormLayerComputerTest {

    @Test
    void compute_on_weight_matrix_and_batch_norm() {
        BatchNormLayer layer = new BatchNormLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = new BatchNormLayerComputer(new MeanDeviationProvider()).computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer).getResult();
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
        DoubleMatrix output = new BatchNormLayerComputer(new MeanDeviationProvider()).computeAFromInput(DataFormatConverter.fromDoubleTabToDoubleMatrix(input), layer).getResult();
        IntStream.range(0, expectedOutput.length).forEach(i -> Assertions.assertArrayEquals(expectedOutput[i], DataFormatConverter.fromDoubleMatrixToDoubleTab(output)[i],0.0001));
    }

}