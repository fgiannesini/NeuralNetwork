package com.fgiannesini.neuralnetwork.learningrate;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class DecayLearningRateUpdaterTest {
    @Test
    void update() {
        ILearningRateUpdater learningRateUpdater = new DecayLearningRateUpdater(0.01);
        Assertions.assertEquals(0.01, learningRateUpdater.get(0), 0.001);
        Assertions.assertEquals(0.005, learningRateUpdater.get(1), 0.001);
        Assertions.assertEquals(0.0017, learningRateUpdater.get(5), 0.001);
    }
}