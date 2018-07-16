package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class RandomInitializerTest {

    @Test
    void initDoubleMatrix() {
        DoubleMatrix initializedMatrix = InitializerType.RANDOM.getInitializer().initDoubleMatrix(5, 2);
        Assertions.assertTrue(Arrays.stream(initializedMatrix.data).allMatch(d -> d < 0.01 && d > 0));
    }
}