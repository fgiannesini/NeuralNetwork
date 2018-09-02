package com.fgiannesini.neuralnetwork.assertions;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.function.Executable;

import java.util.Arrays;
import java.util.List;

public class DoubleMatrixAssertions {
    public static List<Executable> getMatrixAssertions(DoubleMatrix currentMatrix, DoubleMatrix expectedMatrix) {
        return Arrays.asList(
                () -> Assertions.assertEquals(expectedMatrix.getColumns(), currentMatrix.getColumns()),
                () -> Assertions.assertEquals(expectedMatrix.getRows(), currentMatrix.getRows()),
                () -> Assertions.assertArrayEquals(expectedMatrix.data, currentMatrix.data, 0.00001)
        );
    }

    public static void assertMatrices(DoubleMatrix currentMatrix, DoubleMatrix expectedMatrix) {
        Assertions.assertAll(getMatrixAssertions(currentMatrix, expectedMatrix));
    }
}
