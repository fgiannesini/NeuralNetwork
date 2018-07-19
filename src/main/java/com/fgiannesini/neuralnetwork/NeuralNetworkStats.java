package com.fgiannesini.neuralnetwork;

public class NeuralNetworkStats {
    private double learningCost;
    private double testCost;
    private int iterationCount;

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
