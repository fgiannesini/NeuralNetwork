package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.learningrate.ILearningRateUpdater;
import com.fgiannesini.neuralnetwork.model.LayerType;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

public class HyperParameters implements Cloneable, Serializable {

    private int epochCount;
    private int batchSize;
    private int hiddenLayerSize[];
    private ILearningRateUpdater learningRateUpdater;
    private Double momentumCoeff;
    private Double rmsStopCoeff;
    private LayerType layerType;
    private RegularizationCoeffs regularizationCoeffs;
    private int[] convolutionLayers;

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

    public HyperParameters regularizationCoeff(RegularizationCoeffs regularizationCoeffs) {
        this.regularizationCoeffs = regularizationCoeffs;
        return this;
    }

    public HyperParameters layerType(LayerType layerType) {
        this.layerType = layerType;
        return this;
    }

    public HyperParameters convolutionLayers(int[] convolutionLayers) {
        this.convolutionLayers = convolutionLayers;
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

    public LayerType getLayerType() {
        return layerType;
    }

    public RegularizationCoeffs getRegularizationCoeffs() {
        return regularizationCoeffs;
    }

    public int[] getConvolutionLayers() {
        return convolutionLayers;
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
                ", convolutionLayers=" + Arrays.toString(convolutionLayers) +
                ", layerType=" + layerType +
                ", learningRateUpdater=" + learningRateUpdater.getClass().getSimpleName() + " " + learningRateUpdater.get(0) +
                ", momentumCoeff=" + momentumCoeff +
                ", rmsStopCoeff=" + rmsStopCoeff +
                ", regularizationCoeffs=" + regularizationCoeffs +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof HyperParameters)) return false;
        HyperParameters that = (HyperParameters) o;
        return epochCount == that.epochCount &&
                batchSize == that.batchSize &&
                Arrays.equals(hiddenLayerSize, that.hiddenLayerSize) &&
                Objects.equals(learningRateUpdater, that.learningRateUpdater) &&
                Objects.equals(momentumCoeff, that.momentumCoeff) &&
                Objects.equals(rmsStopCoeff, that.rmsStopCoeff) &&
                layerType == that.layerType &&
                Objects.equals(regularizationCoeffs, that.regularizationCoeffs) &&
                Arrays.equals(convolutionLayers, that.convolutionLayers);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(epochCount, batchSize, learningRateUpdater, momentumCoeff, rmsStopCoeff, layerType, regularizationCoeffs);
        result = 31 * result + Arrays.hashCode(hiddenLayerSize);
        result = 31 * result + Arrays.hashCode(convolutionLayers);
        return result;
    }
}
