package com.fgiannesini.neuralnetwork.gradient;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class GradientDescentBuilderTest {

    @Test
    void test_exception_if_neuralNetworkModel_missing() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> GradientDescentBuilder.init().build());
    }

}