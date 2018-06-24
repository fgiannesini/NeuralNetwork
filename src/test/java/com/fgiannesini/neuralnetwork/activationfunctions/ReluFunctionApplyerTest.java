package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class ReluFunctionApplyerTest {

    private ReluFunctionApplyer reluFunctionApplyer;

    @BeforeEach
    void setUp() {
        reluFunctionApplyer = new ReluFunctionApplyer();
    }

    @Test
    void nominal() {
        Assertions.assertAll(
                checkRelu(new float[]{0}, new float[]{0}),
                checkRelu(new float[]{1}, new float[]{1}),
                checkRelu(new float[]{0}, new float[]{-1}),
                checkRelu(new float[]{1f, 2f, 3f, 4f, 5f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkRelu(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, reluFunctionApplyer.apply(new FloatMatrix(input)).data, 0.0001f);
    }
}