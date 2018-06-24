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
    void nominal() {
        Assertions.assertAll(
                checkSigmoid(new float[]{0.5f}, new float[]{0}),
                checkSigmoid(new float[]{0.7310f}, new float[]{1}),
                checkSigmoid(new float[]{0.2689f}, new float[]{-1}),
                checkSigmoid(new float[]{0.7310f, 0.88079f, 0.95257f, 0.98201f, 0.99330f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkSigmoid(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, sigmoidFunctionApplier.apply(new FloatMatrix(input)).data, 0.0001f);
    }
}