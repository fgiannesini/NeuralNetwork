package com.fgiannesini.neuralnetwork.learningrate;

import java.util.Objects;

public class DecayLearningRateUpdater implements ILearningRateUpdater {
    private final double initialLearningRate;

    public DecayLearningRateUpdater(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    @Override
    public double get(int epochNumber) {
        return initialLearningRate / (1 + epochNumber);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DecayLearningRateUpdater)) return false;
        DecayLearningRateUpdater that = (DecayLearningRateUpdater) o;
        return Double.compare(that.initialLearningRate, initialLearningRate) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(initialLearningRate);
    }
}
