package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
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
                checkReluApply(new double[]{0}, new double[]{0}),
                checkReluApply(new double[]{1}, new double[]{1}),
                checkReluApply(new double[]{0}, new double[]{-1}),
                checkReluApply(new double[]{1f, 2f, 3f, 4f, 5f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkReluApply(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, reluFunctionApplier.apply(new DoubleMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkReluDerivate(new double[]{1}, new double[]{0}),
                checkReluDerivate(new double[]{1}, new double[]{2}),
                checkReluDerivate(new double[]{0}, new double[]{-2}),
                checkReluDerivate(new double[]{1f, 1f, 1f, 1f, 1f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkReluDerivate(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, reluFunctionApplier.derivate(new DoubleMatrix(input)).data, 0.0001f);
    }
}