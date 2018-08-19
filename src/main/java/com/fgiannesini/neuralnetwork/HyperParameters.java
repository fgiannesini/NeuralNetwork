package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.learningrate.ILearningRateUpdater;

import java.util.Arrays;

public class HyperParameters implements Cloneable {

    private int epochCount;
    private int batchSize;
    private int hiddenLayerSize[];
    private ILearningRateUpdater learningRateUpdater;
    private Double momentumCoeff;
    private Double rmsStopCoeff;

    public HyperParameters() {
    }

    public HyperParameters epochCount(int epochCount) {
        this.epochCount = epochCount;
        return this;
    }

    public HyperParameters batchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public HyperParameters hiddenLayerSize(int[] hiddenLayerSize) {
        this.hiddenLayerSize = hiddenLayerSize;
        return this;
    }

    public HyperParameters learningRateUpdater(ILearningRateUpdater learningRateUpdater) {
        this.learningRateUpdater = learningRateUpdater;
        return this;
    }

    public HyperParameters momentumCoeff(Double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
        return this;
    }

    public HyperParameters rmsStopCoeff(Double rmsStopCoeff) {
        this.rmsStopCoeff = rmsStopCoeff;
        return this;
    }

    public int getEpochCount() {
        return epochCount;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int[] getHiddenLayerSize() {
        return hiddenLayerSize;
    }

    public ILearningRateUpdater getLearningRateUpdater() {
        return learningRateUpdater;
    }

    public Double getMomentumCoeff() {
        return momentumCoeff;
    }

    public Double getRmsStopCoeff() {
        return rmsStopCoeff;
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
                ", hiddenLayerSize=" + Arrays.toString(hiddenLayerSize) +
                ", learningRateUpdater=" + learningRateUpdater.getClass().getSimpleName() + " " + learningRateUpdater.get(0) +
                ", momentumCoeff=" + momentumCoeff +
                ", rmsStopCoeff=" + rmsStopCoeff +
                '}';
    }
}
