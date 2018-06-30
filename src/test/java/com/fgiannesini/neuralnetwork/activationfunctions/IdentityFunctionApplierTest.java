package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class IdentityFunctionApplierTest {

    private IdentityFunctionApplier identityFunctionApplier;

    @BeforeEach
    void setUp() {
        identityFunctionApplier = new IdentityFunctionApplier();
    }

    @Test
    void apply() {
        Assertions.assertAll(
                checkTanhApply(new double[]{0}, new double[]{0}),
                checkTanhApply(new double[]{1f}, new double[]{1}),
                checkTanhApply(new double[]{-1f}, new double[]{-1}),
                checkTanhApply(new double[]{1f, 2f, 3f, 4f, 5f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhApply(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, identityFunctionApplier.apply(new DoubleMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkTanhDerivate(new double[]{0f}, new double[]{0}),
                checkTanhDerivate(new double[]{2f}, new double[]{2}),
                checkTanhDerivate(new double[]{-2f}, new double[]{-2}),
                checkTanhDerivate(new double[]{1f, 2f, 3f, 4f, 5f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhDerivate(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, identityFunctionApplier.derivate(new DoubleMatrix(input)).data, 0.0001f);
    }
}