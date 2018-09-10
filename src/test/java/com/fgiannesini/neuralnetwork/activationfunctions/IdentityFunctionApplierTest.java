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
                checkTanhApply(new double[]{1}, new double[]{1}),
                checkTanhApply(new double[]{-1}, new double[]{-1}),
                checkTanhApply(new double[]{1, 2, 3, 4, 5}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhApply(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, identityFunctionApplier.apply(new DoubleMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkIdentityDerivate(new double[]{1}, new double[]{0}),
                checkIdentityDerivate(new double[]{1}, new double[]{2}),
                checkIdentityDerivate(new double[]{1}, new double[]{-2}),
                checkIdentityDerivate(new double[]{1, 1, 1, 1, 1}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkIdentityDerivate(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, identityFunctionApplier.derivate(new DoubleMatrix(input), null).data, 0.0001f);
    }
}