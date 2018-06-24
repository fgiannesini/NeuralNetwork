package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
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
    void nominal() {
        Assertions.assertAll(
                checkTanh(new float[]{0}, new float[]{0}),
                checkTanh(new float[]{0.7615f}, new float[]{1}),
                checkTanh(new float[]{-0.7615f}, new float[]{-1}),
                checkTanh(new float[]{0.7615f, 0.9640f, 0.9950f, 0.9993f, 0.9999f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanh(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, tanhFunctionApplier.apply(new FloatMatrix(input)).data, 0.0001f);
    }
}