package com.fgiannesini.neuralnetwork.assertions;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;

import java.util.List;
import java.util.stream.IntStream;

public class DoubleMatrixAssertions {

    public static void assertMatrices(DoubleMatrix currentMatrix, DoubleMatrix expectedMatrix) {
        Assertions.assertEquals(expectedMatrix.getColumns(), currentMatrix.getColumns());
        Assertions.assertEquals(expectedMatrix.getRows(), currentMatrix.getRows());
        Assertions.assertArrayEquals(expectedMatrix.data, currentMatrix.data, 0.00001);
    }

    public static void assertMatrices(List<DoubleMatrix> currentMatrix, List<DoubleMatrix> expectedMatrix) {
        Assertions.assertEquals(currentMatrix.size(), expectedMatrix.size());
        IntStream.range(0, currentMatrix.size()).forEach(i -> Assertions.assertEquals(expectedMatrix.get(i), currentMatrix.get(i)));
    }
}
