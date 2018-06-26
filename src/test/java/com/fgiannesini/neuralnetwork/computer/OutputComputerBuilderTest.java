package com.fgiannesini.neuralnetwork.computer;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class OutputComputerBuilderTest {

    @Test
    void test_exception_if_neuralNetworkModel_missing() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> OutputComputerBuilder.init().build());
    }

}