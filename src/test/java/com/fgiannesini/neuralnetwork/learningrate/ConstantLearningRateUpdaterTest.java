package com.fgiannesini.neuralnetwork.learningrate;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class ConstantLearningRateUpdaterTest {

    @Test
    void update() {
        ILearningRateUpdater learningRateUpdater = new ConstantLearningRateUpdater(0.01);
        Assertions.assertEquals(0.01, learningRateUpdater.get(0), 0.01);
        Assertions.assertEquals(0.01, learningRateUpdater.get(1), 0.01);
        Assertions.assertEquals(0.01, learningRateUpdater.get(5), 0.01);
    }
}