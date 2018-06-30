package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class SigmoidFunctionApplierTest {

    private SigmoidFunctionApplier sigmoidFunctionApplier;

    @BeforeEach
    void setUp() {
        sigmoidFunctionApplier = new SigmoidFunctionApplier();
    }

    @Test
    void apply() {
        Assertions.assertAll(
                checkSigmoidApply(new double[]{0.5f}, new double[]{0}),
                checkSigmoidApply(new double[]{0.7310f}, new double[]{1}),
                checkSigmoidApply(new double[]{0.2689f}, new double[]{-1}),
                checkSigmoidApply(new double[]{0.7310f, 0.88079f, 0.95257f, 0.98201f, 0.99330f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkSigmoidApply(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.apply(new DoubleMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkSigmoidDerivate(new double[]{0f}, new double[]{0}),
                checkSigmoidDerivate(new double[]{-2f}, new double[]{2}),
                checkSigmoidDerivate(new double[]{-6f}, new double[]{-2}),
                checkSigmoidDerivate(new double[]{0f, -2f, -6f, -12f, -20f}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkSigmoidDerivate(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.derivate(new DoubleMatrix(input)).data, 0.0001f);
    }
}