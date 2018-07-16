package com.fgiannesini.neuralnetwork.initializer;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class UniformInitializerTest {

    @Test
    void initDoubleMatrix() {
        DoubleMatrix initializedMatrix = InitializerType.UNIFORM.getInitializer().initDoubleMatrix(5, 2);
        Assertions.assertTrue(Arrays.stream(initializedMatrix.data).allMatch(d -> d > 0 && d < 0.4472));
    }

}