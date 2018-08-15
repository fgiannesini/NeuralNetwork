package com.fgiannesini.neuralnetwork.learningrate;

public class ConstantLearningRateUpdater implements ILearningRateUpdater {

    private final double learningRate;

    public ConstantLearningRateUpdater(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public double get(int epochNumber) {
        return learningRate;
    }
}
