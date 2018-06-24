package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class LeakyReluFunctionApplierTest {

    private LeakyReluFunctionApplier sigmoidFunctionApplier;

    @BeforeEach
    void setUp() {
        sigmoidFunctionApplier = new LeakyReluFunctionApplier();
    }

    @Test
    void nominal() {
        Assertions.assertAll(
                checkLeakyRelu(new float[]{0f}, new float[]{0}),
                checkLeakyRelu(new float[]{1f}, new float[]{1}),
                checkLeakyRelu(new float[]{-0.1f}, new float[]{-1}),
                checkLeakyRelu(new float[]{1f, 2f, 3f, 4f, 5f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkLeakyRelu(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.apply(new FloatMatrix(input)).data, 0.0001f);
    }
}