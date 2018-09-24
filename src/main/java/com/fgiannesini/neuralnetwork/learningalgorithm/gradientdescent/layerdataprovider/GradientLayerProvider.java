package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import org.jblas.DoubleMatrix;

import java.util.List;

public abstract class GradientLayerProvider {

    private final List<DoubleMatrix> results;
    int currentLayerIndex;

    GradientLayerProvider(List<DoubleMatrix> results, int layersCount) {
        currentLayerIndex = layersCount;
        this.results = results;
    }

    public void nextLayer() {
        currentLayerIndex--;
    }

    public boolean hasNextLayer() {
        return currentLayerIndex > 0;
    }

    public DoubleMatrix getPreviousResult() {
        return results.get(currentLayerIndex - 1);
    }

    public DoubleMatrix getCurrentResult() {
        return results.get(currentLayerIndex);
    }

    public int getCurrentLayerIndex() {
        return currentLayerIndex;
    }

}
