package com.fgiannesini.neuralnetwork;

public class HyperParameters {

    private int iterationCount;
    private int batchSize;
    private int inputCount;
    private int testInputCount;
    private int hiddenLayerSize[];

    public HyperParameters() {
        this.iterationCount = 1;
        this.batchSize = 1000;
        this.inputCount = 100_000;
        this.testInputCount = 100;

        this.hiddenLayerSize = new int[]{10};
    }

    public HyperParameters iterationCount(int iterationCount) {
        this.iterationCount = iterationCount;
        return this;
    }

    public HyperParameters batchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public HyperParameters inputCount(int inputCount) {
        this.inputCount = inputCount;
        return this;
    }

    public HyperParameters tInputCount(int testInputCount) {
        this.testInputCount = testInputCount;
        return this;
    }

    public HyperParameters hiddenLayerSize(int[] hiddenLayerSize) {
        this.hiddenLayerSize = hiddenLayerSize;
        return this;
    }

    public int getIterationCount() {
        return iterationCount;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getInputCount() {
        return inputCount;
    }

    public int getTestInputCount() {
        return testInputCount;
    }

    public int[] getHiddenLayerSize() {
        return hiddenLayerSize;
    }
}
