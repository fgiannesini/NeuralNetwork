package com.fgiannesini.neuralnetwork.learningrate;

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
}
