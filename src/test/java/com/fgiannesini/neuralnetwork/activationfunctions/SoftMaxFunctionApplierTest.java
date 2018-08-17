package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class SoftMaxFunctionApplierTest {

    private SoftMaxFunctionApplier softMaxFunctionApplier;

    @BeforeEach
    void setUp() {
        softMaxFunctionApplier = new SoftMaxFunctionApplier();
    }

    @Test
    void apply() {
        Assertions.assertAll(
                checkSoftMaxApply(new DoubleMatrix(2, 2, 0.2689, 0.7310, 0.1192, 0.8807), new DoubleMatrix(2, 2, -1, 0, 2, 4))
        );
    }

    private Executable checkSoftMaxApply(DoubleMatrix expected, DoubleMatrix input) {
        return () -> Assertions.assertArrayEquals(expected.data, softMaxFunctionApplier.apply(input).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkSoftMaxDerivate(new double[]{0}, new double[]{0}),
                checkSoftMaxDerivate(new double[]{-2}, new double[]{2}),
                checkSoftMaxDerivate(new double[]{-6}, new double[]{-2}),
                checkSoftMaxDerivate(new double[]{0, -2, -6, -12, -20}, new double[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkSoftMaxDerivate(double[] expected, double[] input) {
        return () -> Assertions.assertArrayEquals(expected, softMaxFunctionApplier.derivate(new DoubleMatrix(input)).data, 0.0001f);
    }

}