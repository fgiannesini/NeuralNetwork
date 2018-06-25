package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class ReluFunctionApplierTest {

    private ReluFunctionApplier reluFunctionApplier;

    @BeforeEach
    void setUp() {
        reluFunctionApplier = new ReluFunctionApplier();
    }

    @Test
    void apply() {
        Assertions.assertAll(
                checkReluApply(new float[]{0}, new float[]{0}),
                checkReluApply(new float[]{1}, new float[]{1}),
                checkReluApply(new float[]{0}, new float[]{-1}),
                checkReluApply(new float[]{1f, 2f, 3f, 4f, 5f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkReluApply(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, reluFunctionApplier.apply(new FloatMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkReluDerivate(new float[]{1}, new float[]{0}),
                checkReluDerivate(new float[]{1}, new float[]{2}),
                checkReluDerivate(new float[]{0}, new float[]{-2}),
                checkReluDerivate(new float[]{1f, 1f, 1f, 1f, 1f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkReluDerivate(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, reluFunctionApplier.derivate(new FloatMatrix(input)).data, 0.0001f);
    }
}