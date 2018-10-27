package com.fgiannesini.neuralnetwork.learningrate;

import java.util.Objects;

public class ExponentiallyLearningRateUpdater implements ILearningRateUpdater {
    private final double initialLearningRate;

    public ExponentiallyLearningRateUpdater(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    @Override
    public double get(int epochNumber) {
        return Math.pow(0.95, epochNumber) * initialLearningRate;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ExponentiallyLearningRateUpdater)) return false;
        ExponentiallyLearningRateUpdater that = (ExponentiallyLearningRateUpdater) o;
        return Double.compare(that.initialLearningRate, initialLearningRate) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(initialLearningRate);
    }
}
