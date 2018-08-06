package com.fgiannesini.neuralnetwork;

public class NeuralNetworkStats {
    private final double learningCost;
    private final double testCost;
    private final int epochNumber;
    private final int batchNumber;

    NeuralNetworkStats(double learningCost, double testCost, int batchNumber, int epochNumber) {
        this.learningCost = learningCost;
        this.testCost = testCost;
        this.batchNumber = batchNumber;
        this.epochNumber = epochNumber;
    }

    public double getLearningCost() {
        return learningCost;
    }

    public double getTestCost() {
        return testCost;
    }

    public int getEpochNumber() {
        return epochNumber;
    }

    public int getBatchNumber() {
        return batchNumber;
    }
}
