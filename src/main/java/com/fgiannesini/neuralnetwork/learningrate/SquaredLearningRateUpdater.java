package com.fgiannesini.neuralnetwork.learningrate;

import java.util.Objects;

public class SquaredLearningRateUpdater implements ILearningRateUpdater {
    private final double initialLearningRate;

    public SquaredLearningRateUpdater(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    @Override
    public double get(int epochNumber) {
        if (epochNumber == 0) {
            return initialLearningRate;
        }
        return initialLearningRate / Math.sqrt(epochNumber);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SquaredLearningRateUpdater)) return false;
        SquaredLearningRateUpdater that = (SquaredLearningRateUpdater) o;
        return Double.compare(that.initialLearningRate, initialLearningRate) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(initialLearningRate);
    }
}
