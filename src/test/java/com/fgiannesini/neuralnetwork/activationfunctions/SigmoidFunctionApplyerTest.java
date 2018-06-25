package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
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
                checkSigmoidApply(new float[]{0.5f}, new float[]{0}),
                checkSigmoidApply(new float[]{0.7310f}, new float[]{1}),
                checkSigmoidApply(new float[]{0.2689f}, new float[]{-1}),
                checkSigmoidApply(new float[]{0.7310f, 0.88079f, 0.95257f, 0.98201f, 0.99330f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkSigmoidApply(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.apply(new FloatMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkSigmoidDerivate(new float[]{0f}, new float[]{0}),
                checkSigmoidDerivate(new float[]{-2f}, new float[]{2}),
                checkSigmoidDerivate(new float[]{-6f}, new float[]{-2}),
                checkSigmoidDerivate(new float[]{0f, -2f, -6f, -12f, -20f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkSigmoidDerivate(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.derivate(new FloatMatrix(input)).data, 0.0001f);
    }
}