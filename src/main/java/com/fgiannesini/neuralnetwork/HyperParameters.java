package com.fgiannesini.neuralnetwork;

import java.util.Arrays;

public class HyperParameters implements Cloneable {

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

    public HyperParameters testInputCount(int testInputCount) {
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

    @Override
    public HyperParameters clone() {
        try {
            return (HyperParameters) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return "HyperParameters{" +
                "iterationCount=" + iterationCount +
                ", batchSize=" + batchSize +
                ", inputCount=" + inputCount +
                ", testInputCount=" + testInputCount +
                ", hiddenLayerSize=" + Arrays.toString(hiddenLayerSize) +
                '}';
    }
}
