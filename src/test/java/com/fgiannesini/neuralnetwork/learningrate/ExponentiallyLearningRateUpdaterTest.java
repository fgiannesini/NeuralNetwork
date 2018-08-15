package com.fgiannesini.neuralnetwork.learningrate;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class ExponentiallyLearningRateUpdaterTest {

    @Test
    void update() {
        ILearningRateUpdater learningRateUpdater = new ExponentiallyLearningRateUpdater(0.01);
        Assertions.assertEquals(0.01, learningRateUpdater.get(0), 0.001);
        Assertions.assertEquals(0.0095, learningRateUpdater.get(1), 0.001);
        Assertions.assertEquals(0.0077, learningRateUpdater.get(5), 0.001);
    }

}