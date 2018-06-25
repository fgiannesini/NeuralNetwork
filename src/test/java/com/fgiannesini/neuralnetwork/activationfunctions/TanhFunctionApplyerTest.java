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
    void apply() {
        Assertions.assertAll(
                checkTanhApply(new float[]{0}, new float[]{0}),
                checkTanhApply(new float[]{0.7615f}, new float[]{1}),
                checkTanhApply(new float[]{-0.7615f}, new float[]{-1}),
                checkTanhApply(new float[]{0.7615f, 0.9640f, 0.9950f, 0.9993f, 0.9999f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhApply(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, tanhFunctionApplier.apply(new FloatMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkTanhDerivate(new float[]{1f}, new float[]{0}),
                checkTanhDerivate(new float[]{-3f}, new float[]{2}),
                checkTanhDerivate(new float[]{-3f}, new float[]{-2}),
                checkTanhDerivate(new float[]{0f, -3f, -8f, -15f, -24f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhDerivate(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, tanhFunctionApplier.derivate(new FloatMatrix(input)).data, 0.0001f);
    }
}