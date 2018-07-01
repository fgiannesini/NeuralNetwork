package com.fgiannesini.neuralnetwork.learningalgorithm;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LearningAlgorithmBuilderTest {

    @Test
    void test_exception_if_neuralNetworkModel_is_missing() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> LearningAlgorithmBuilder.init().build());
    }

}