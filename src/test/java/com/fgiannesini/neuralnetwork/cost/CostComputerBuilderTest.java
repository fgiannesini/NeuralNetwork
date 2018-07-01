package com.fgiannesini.neuralnetwork.cost;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CostComputerBuilderTest {

    @Test
    void check_neural_network_is_mandatory() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> CostComputerBuilder.init().build());
    }

}