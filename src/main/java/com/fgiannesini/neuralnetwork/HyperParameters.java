package com.fgiannesini.neuralnetwork;

import java.util.Arrays;

public class HyperParameters implements Cloneable {

    private int epochCount;
    private int batchSize;
    private int inputCount;
    private int testInputCount;
    private int hiddenLayerSize[];

    public HyperParameters() {
        this.epochCount = 1;
        this.batchSize = 1000;
        this.inputCount = 100_000;
        this.testInputCount = 100;

        this.hiddenLayerSize = new int[]{10};
    }

    public HyperParameters epochCount(int epochCount) {
        this.epochCount = epochCount;
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

    public int getEpochCount() {
        return epochCount;
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
                "epochCount=" + epochCount +
                ", batchSize=" + batchSize +
                ", inputCount=" + inputCount +
                ", testInputCount=" + testInputCount +
                ", hiddenLayerSize=" + Arrays.toString(hiddenLayerSize) +
                '}';
    }
}
