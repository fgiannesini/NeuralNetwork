package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

class GradientLayerProvider {

    private final List<DoubleMatrix> results;
    private final List<Layer> layers;
    private int currentLayerIndex;

    public GradientLayerProvider(List<Layer> layers) {
        currentLayerIndex = layers.size();
        this.results = new ArrayList<>(layers.size());
        this.layers = layers;
    }

    void addGradientLayerResult(DoubleMatrix result) {
        results.add(result);
    }

    public void nextLayer() {
        currentLayerIndex--;
    }

    public boolean hasNextLayer() {
        return currentLayerIndex > 0;
    }

    public DoubleMatrix getPreviousWeightMatrix() {
        return layers.get(currentLayerIndex).getWeightMatrix();
    }

    public ActivationFunctionApplier getCurrentActivationFunction() {
        return layers.get(currentLayerIndex - 1).getActivationFunctionType().getActivationFunction();
    }

    public DoubleMatrix getPreviousResult() {
        return results.get(currentLayerIndex - 1);
    }

    public DoubleMatrix getCurrentResult() {
        return results.get(currentLayerIndex);
    }

}
