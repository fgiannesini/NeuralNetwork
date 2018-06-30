package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class TanhFunctionApplierTest {

    private TanhFunctionApplier tanhFunctionApplier;

    @BeforeEach
    void setUp() {
        tanhFunctionApplier = new TanhFunctionApplier();
    }

    @Test
    void apply() {
        Assertions.assertAll(
                checkTanhApply(new double[]{0}, new double[]{0}),
                checkTanhApply(new double[]{0.7615f}, new double[]{1}),
                checkTanhApply(new double[]{-0.7615f}, new double[]{-1}),
                checkTanhApply(new double[]{0.7615f, 0.9640f, 0.9950f, 0.9993f, 0.9999f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhApply(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, tanhFunctionApplier.apply(new DoubleMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkTanhDerivate(new double[]{1f}, new double[]{0}),
                checkTanhDerivate(new double[]{-3f}, new double[]{2}),
                checkTanhDerivate(new double[]{-3f}, new double[]{-2}),
                checkTanhDerivate(new double[]{0f, -3f, -8f, -15f, -24f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhDerivate(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, tanhFunctionApplier.derivate(new DoubleMatrix(input)).data, 0.0001f);
    }
}