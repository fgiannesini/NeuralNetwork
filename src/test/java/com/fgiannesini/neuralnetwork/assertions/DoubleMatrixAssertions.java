package com.fgiannesini.neuralnetwork.assertions;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;

public class DoubleMatrixAssertions {

    public static void assertMatrices(DoubleMatrix currentMatrix, DoubleMatrix expectedMatrix) {
        Assertions.assertEquals(expectedMatrix.getColumns(), currentMatrix.getColumns());
        Assertions.assertEquals(expectedMatrix.getRows(), currentMatrix.getRows());
        Assertions.assertArrayEquals(expectedMatrix.data, currentMatrix.data, 0.00001);
    }
}
