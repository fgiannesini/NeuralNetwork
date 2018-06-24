package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

class TanhFunctionApplyerTest {

    private TanhFunctionApplyer tanhFunctionApplyer;

    @BeforeEach
    void setUp() {
        tanhFunctionApplyer = new TanhFunctionApplyer();
    }

    @Test
    void nominal() {
        Assertions.assertAll(
                checkSigmoid(new float[]{0}, new float[]{0}),
                checkSigmoid(new float[]{0.7615f}, new float[]{1}),
                checkSigmoid(new float[]{-0.7615f}, new float[]{-1}),
                checkSigmoid(new float[]{0.7615f, 0.9640f, 0.9950f, 0.9993f, 0.9999f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkSigmoid(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, tanhFunctionApplyer.apply(new FloatMatrix(input)).data, 0.0001f);
    }
}