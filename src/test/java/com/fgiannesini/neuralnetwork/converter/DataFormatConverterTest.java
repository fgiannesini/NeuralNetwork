package com.fgiannesini.neuralnetwork.converter;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

class DataFormatConverterTest {

    @Test
    void convert_tab_to_DoubleMatrix() {
        DoubleMatrix result = DataFormatConverter.fromTabToDoubleMatrix(new double[]{0, 1, 2});
        Assertions.assertEquals(3, result.getRows());
        Assertions.assertEquals(1, result.getColumns());
        Assertions.assertArrayEquals(new double[]{0, 1, 2}, result.data, 0.0001);
    }

    @Test
    void convert_double_tab_to_DoubleMatrix() {
        DoubleMatrix result = DataFormatConverter.fromDoubleTabToDoubleMatrix(new double[][]{{0, 1, 2}, {3, 4, 5}});
        Assertions.assertEquals(3, result.getRows());
        Assertions.assertEquals(2, result.getColumns());
        Assertions.assertArrayEquals(new double[]{0, 1, 2, 3, 4, 5}, result.data, 0.0001);
    }

    @Test
    void fromDoubleMatrixToTab() {
        DoubleMatrix input = new DoubleMatrix(3, 1, 0, 1, 2);
        double[] result = DataFormatConverter.fromDoubleMatrixToTab(input);
        Assertions.assertArrayEquals(new double[]{0, 1, 2}, result, 0.0001);
    }

    @Test
    void fromDoubleMatrixToDoubleTab() {
        DoubleMatrix input = new DoubleMatrix(3, 2, 0, 1, 2, 3, 4, 5);
        double[][] result = DataFormatConverter.fromDoubleMatrixToDoubleTab(input);
        double[][] expected = {{0, 1, 2}, {3, 4, 5}};
        Assertions.assertEquals(expected.length, result.length);
        IntStream.range(0, expected.length).forEach(index -> {
            Assertions.assertArrayEquals(expected[index], result[index], 0.0001);
        });
    }
}