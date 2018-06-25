package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
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
                checkTanhApply(new float[]{0}, new float[]{0}),
                checkTanhApply(new float[]{1f}, new float[]{1}),
                checkTanhApply(new float[]{-1f}, new float[]{-1}),
                checkTanhApply(new float[]{1f, 2f, 3f, 4f, 5f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhApply(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, identityFunctionApplier.apply(new FloatMatrix(input)).data, 0.0001f);
    }

    @Test
    void derivate() {
        Assertions.assertAll(
                checkTanhDerivate(new float[]{0f}, new float[]{0}),
                checkTanhDerivate(new float[]{2f}, new float[]{2}),
                checkTanhDerivate(new float[]{-2f}, new float[]{-2}),
                checkTanhDerivate(new float[]{1f, 2f, 3f, 4f, 5f}, new float[]{1, 2, 3, 4, 5})
        );
    }

    private Executable checkTanhDerivate(float[] expected, float[] input) {
        return () -> Assertions.assertArrayEquals(expected, identityFunctionApplier.derivate(new FloatMatrix(input)).data, 0.0001f);
    }
}