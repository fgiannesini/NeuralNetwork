package com.fgiannesini.neuralnetwork;

public class NeuralNetworkStats {
    private final double learningCost;
    private final double testCost;
    private final int iterationNumber;
    private int batchNumber;

    NeuralNetworkStats(double learningCost, double testCost, int batchNumber, int iterationNumber) {
        this.learningCost = learningCost;
        this.testCost = testCost;
        this.batchNumber = batchNumber;
        this.iterationNumber = iterationNumber;
    }

    public double getLearningCost() {
        return learningCost;
    }

    public double getTestCost() {
        return testCost;
    }

    public int getIterationNumber() {
        return iterationNumber;
    }

    public int getBatchNumber() {
        return batchNumber;
    }
}
