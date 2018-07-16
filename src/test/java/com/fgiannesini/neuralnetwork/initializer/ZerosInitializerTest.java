package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class ZerosInitializerTest {

    @Test
    void initDoubleMatrix() {
        DoubleMatrix initializedMatrix = InitializerType.ZEROS.getInitializer().initDoubleMatrix(5, 2);
        Assertions.assertTrue(Arrays.stream(initializedMatrix.data).allMatch(d -> d == 0));
    }
}