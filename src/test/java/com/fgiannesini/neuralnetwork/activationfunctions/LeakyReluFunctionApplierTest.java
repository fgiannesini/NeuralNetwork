package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
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
    void apply() {
        Assertions.assertAll(
                checkLeakyReluApply(new double[]{0f}, new double[]{0}),
                checkLeakyReluApply(new double[]{1f}, new double[]{1}),
                checkLeakyReluApply(new double[]{-0.01f}, new double[]{-1}),
                checkLeakyReluApply(new double[]{1f, 2f, 3f, 4f, 5f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkLeakyReluApply(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.apply(new DoubleMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkLeakyReluDerivate(new double[]{1f}, new double[]{0}),
                checkLeakyReluDerivate(new double[]{1f}, new double[]{2}),
                checkLeakyReluDerivate(new double[]{-0.01f}, new double[]{-2}),
                checkLeakyReluDerivate(new double[]{1f, 1f, 1f, 1f, 1f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkLeakyReluDerivate(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.derivate(new DoubleMatrix(input)).data, 0.0001f);
    }
}