package com.fgiannesini.neuralnetwork;

public class NeuralNetworkStats {
    private final double learningCost;
    private final double testCost;
    private final int iterationCount;

    NeuralNetworkStats(double learningCost, double testCost, int iterationCount) {
        this.learningCost = learningCost;
        this.testCost = testCost;
        this.iterationCount = iterationCount;
    }

    public double getLearningCost() {
        return learningCost;
    }

    public double getTestCost() {
        return testCost;
    }

    public int getIterationCount() {
        return iterationCount;
    }
}
