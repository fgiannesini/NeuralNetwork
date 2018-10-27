package com.fgiannesini.neuralnetwork.learningrate;

import java.util.Objects;

public class ConstantLearningRateUpdater implements ILearningRateUpdater {

    private final double learningRate;

    public ConstantLearningRateUpdater(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public double get(int epochNumber) {
        return learningRate;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ConstantLearningRateUpdater)) return false;
        ConstantLearningRateUpdater that = (ConstantLearningRateUpdater) o;
        return Double.compare(that.learningRate, learningRate) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(learningRate);
    }
}
