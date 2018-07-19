package com.fgiannesini.neuralnetwork;

public class NeuralNetworkStats {
    private double learningCost;
    private double testCost;

    NeuralNetworkStats(double learningCost, double testCost) {
        this.learningCost = learningCost;
        this.testCost = testCost;
    }

    public double getLearningCost() {
        return learningCost;
    }

    public double getTestCost() {
        return testCost;
    }
}
