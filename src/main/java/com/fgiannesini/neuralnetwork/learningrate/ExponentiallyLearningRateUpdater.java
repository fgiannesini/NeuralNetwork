package com.fgiannesini.neuralnetwork.learningrate;

public class ExponentiallyLearningRateUpdater implements ILearningRateUpdater {
    private final double initialLearningRate;

    public ExponentiallyLearningRateUpdater(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    @Override
    public double get(int epochNumber) {
        return Math.pow(0.95, epochNumber) * initialLearningRate;
    }
}
