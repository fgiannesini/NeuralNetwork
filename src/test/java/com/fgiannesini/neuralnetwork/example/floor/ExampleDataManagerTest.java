package com.fgiannesini.neuralnetwork.example.floor;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class ExampleDataManagerTest {

    @Test
    void GenerateInputData() {
        DoubleMatrix input = ExampleDataManager.generateInputData(12);
        Assertions.assertEquals(12, input.columns);
        Assertions.assertEquals(1, input.rows);
        Assertions.assertTrue(Arrays.stream(input.data).allMatch(d -> d >= 0 && d < 10));
    }

    @Test
    void convertToOutputFormat_input_is_int_tab() {
        DoubleMatrix outputDataMatrix = ExampleDataManager.convertToOutputFormat(new int[]{0, 5, 8});
        double[][] outputData = DataFormatConverter.fromDoubleMatrixToDoubleTab(outputDataMatrix);
        Assertions.assertEquals(3, outputData.length);
        Assertions.assertArrayEquals(outputData[0], new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        Assertions.assertArrayEquals(outputData[1], new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0});
        Assertions.assertArrayEquals(outputData[2], new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
    }

    @Test
    void convertToOutputFormat_input_is_double_tab() {
        DoubleMatrix outputDataMatrix = ExampleDataManager.convertToOutputFormat(new double[]{0.9654, 5.458, 8.487});
        double[][] outputData = DataFormatConverter.fromDoubleMatrixToDoubleTab(outputDataMatrix);
        Assertions.assertEquals(3, outputData.length);
        Assertions.assertArrayEquals(outputData[0], new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        Assertions.assertArrayEquals(outputData[1], new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0});
        Assertions.assertArrayEquals(outputData[2], new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
    }

    @Test
    void computeSuccessRate_perfect() {
        double[][] expected = new double[][]{
                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}
        };

        double[][] predicted = new double[][]{
                {0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0},
                {0.1, 0.2, 0, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4},
                {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0}
        };
        double successRate = ExampleDataManager.computeSuccessRate(DataFormatConverter.fromDoubleTabToDoubleMatrix(expected), DataFormatConverter.fromDoubleTabToDoubleMatrix(predicted));
        Assertions.assertEquals(100, successRate, 0.01);
    }

    @Test
    void computeSuccessRate() {
        double[][] expected = new double[][]{
                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}
        };

        double[][] predicted = new double[][]{
                {0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0},
                {0.9, 0.2, 0, 0.3, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4},
                {0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0}
        };
        double successRate = ExampleDataManager.computeSuccessRate(DataFormatConverter.fromDoubleTabToDoubleMatrix(expected), DataFormatConverter.fromDoubleTabToDoubleMatrix(predicted));
        Assertions.assertEquals(33.33, successRate, 0.01);
    }
}