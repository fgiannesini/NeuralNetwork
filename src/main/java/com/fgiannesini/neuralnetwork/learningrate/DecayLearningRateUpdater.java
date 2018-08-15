package com.fgiannesini.neuralnetwork.learningrate;

public class DecayLearningRateUpdater implements ILearningRateUpdater {
    private final double initialLearningRate;

    public DecayLearningRateUpdater(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    @Override
    public double get(int epochNumber) {
        return initialLearningRate / (1 + epochNumber);
    }
}
