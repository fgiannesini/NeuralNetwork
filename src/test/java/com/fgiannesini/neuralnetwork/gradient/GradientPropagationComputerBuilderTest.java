package com.fgiannesini.neuralnetwork.gradient;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class GradientPropagationComputerBuilderTest {

    @Test
    void test_exception_if_neuralNetworkModel_missing() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> GradientPropagationComputerBuilder.init().build());
    }

}